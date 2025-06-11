# train.py


import os
import pickle
import argparse
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import time

from alphazero.model import AlphaZeroNet
from alphazero.move_encoder import MoveEncoder

from automate import format_time

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
REPLAY_DIR = "replay_buffer" # Directory where self-play games are stored
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# The number of game positions to sample from the replay buffer for each training run.
TRAIN_WINDOW_SIZE = 100_000

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Class & Data Loading
# ─────────────────────────────────────────────────────────────────────────────

class AZDataset(Dataset):
    """
    Custom PyTorch Dataset with data augmentation.
    With a 50% chance for each item, it will horizontally flip the board state
    and policy vector.
    """
    def __init__(self, examples, encoder: MoveEncoder):
        self.examples = examples
        self.encoder = encoder

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state_np, pi_arr, z = self.examples[idx]
        
        # Data Augmentation: 50% chance to flip horizontally
        if random.random() < 0.5:
            # Flip the board state tensor along the file (width) axis
            state_np = np.ascontiguousarray(np.flip(state_np, axis=2))
            
            # Flip the policy vector using the pre-computed map from the MoveEncoder
            original_pi = torch.from_numpy(pi_arr)
            flipped_pi = torch.zeros_like(original_pi)
            for i, prob in enumerate(original_pi):
                if prob > 0:
                    flipped_idx = self.encoder.flip_map[i]
                    flipped_pi[flipped_idx] = prob
            pi_arr = flipped_pi.numpy()

        x = torch.from_numpy(state_np).float()
        pi = torch.from_numpy(pi_arr).float()
        z_tensor = torch.tensor([z], dtype=torch.float32)
        return x, pi, z_tensor

def load_training_window(encoder: MoveEncoder):
    """
    Loads a random subset of games from the replay buffer directory.
    This prevents memory issues with very large replay buffers on disk.
    """
    print(f"Searching for game files in '{REPLAY_DIR}'...", flush=True)
    game_files = [os.path.join(REPLAY_DIR, f) for f in os.listdir(REPLAY_DIR) if f.endswith('.pkl')]
    
    if not game_files:
        print("No game files found. Cannot train.", flush=True)
        return None

    print(f"Found {len(game_files)} total games. Loading a random window of ~{TRAIN_WINDOW_SIZE} positions...", flush=True)
    all_examples = []
    # Keep loading random games until we hit our window size
    while len(all_examples) < TRAIN_WINDOW_SIZE and game_files:
        file_path = random.choice(game_files)
        game_files.remove(file_path) # Avoid picking the same game twice
        try:
            with open(file_path, "rb") as f:
                all_examples.extend(pickle.load(f))
        except Exception as e:
            print(f"Warning: Could not load or process {file_path}. Error: {e}", flush=True)

    if not all_examples:
        print("Failed to load any valid examples from the files sampled. Cannot train.", flush=True)
        return None
        
    print(f"Loaded {len(all_examples)} positions for this training run.", flush=True)
    return AZDataset(all_examples, encoder)

# ─────────────────────────────────────────────────────────────────────────────
#  Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train(net, device, train_loader, val_loader, config):
    """
    The main training and validation loop for the model.
    It now takes a single 'config' object for all hyperparameters.
    """
    is_cuda = (device == "cuda")
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    total_start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        epoch_start_time = time.time()
        # --- Training Phase ---
        net.train()
        total_train_loss, train_batches = 0.0, 0
        for x, pi, z in train_loader:
            x, pi, z = x.to(device), pi.to(device), z.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=is_cuda):
                logits, v = net(x)
                loss = F.cross_entropy(logits, pi) + F.mse_loss(v.squeeze(), z.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            train_batches += 1
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0
        
        # --- Validation Phase ---
        net.eval()
        total_val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for x, pi, z in val_loader:
                x, pi, z = x.to(device), pi.to(device), z.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=is_cuda):
                    logits, v = net(x)
                    loss = F.cross_entropy(logits, pi) + F.mse_loss(v.squeeze(), z.squeeze())
                total_val_loss += loss.item()
                val_batches += 1
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{config.max_epochs} | Val Loss: {avg_val_loss:.4f} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.1e}", flush=True)
        print(f"Time: {format_time(time.time() - epoch_start_time)}", flush=True)
        wandb.log({"epoch": epoch, "val_loss": avg_val_loss, "train_loss": avg_train_loss, "learning_rate": current_lr})
        
        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, "candidate.pth"))
            print(f"  Validation loss improved. Saved new candidate model.", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"  Stopping early as validation loss has not improved for {config.patience} epochs.", flush=True)
                break
    
    # --- Artifact Logging ---
    if best_epoch != -1:
        print("--- Logging best model to wandb artifacts ---", flush=True)
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", # Name the artifact after the unique run ID
            type="model",
            metadata={"best_epoch": best_epoch, "validation_loss": best_val_loss}
        )
        artifact.add_file(os.path.join(CHECKPOINT_DIR, "candidate.pth"))
        wandb.run.log_artifact(artifact)
    
    print(f"Training session complete. Time: {format_time(time.time() - total_start_time)}. Logged summary to wandb.", flush=True)

# --- Main Execution Block ---
def main():
    """
    This script's main function is designed to be called by `wandb agent`.
    It defines all hyperparameters with their defaults. The agent will override
    any of these that are defined in the sweep configuration.
    """
    parser = argparse.ArgumentParser()
    # Define ALL parameters here. The agent will pass values for the ones being swept.
    # The others will correctly use their default values.
    parser.add_argument("--lr", type=float, default=0.007135)
    parser.add_argument("--weight_decay", type=float, default=0.00022968)
    parser.add_argument("--batch_size", type=int, default=1536) # Your fixed batch size
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-wandb", action="store_true") # For local testing
    parser.add_argument("--gen-id", type=str, help="Generation ID for this run. Required in pipeline mode, optional in sweep mode.")
    args = parser.parse_args()

    # Initialize wandb with the new format
    if args.gen_id:
        # Pipeline mode: Use the provided generation ID
        wandb.init(
            project="alphazero-chess",
            group=args.gen_id,
            name=f"{args.gen_id}-training",
            job_type="training",
            config=args,  # Pass argparse values to wandb for sweeps
            mode="disabled" if args.no_wandb else "online"
        )
    else:
        # Sweep mode: Create a standalone run
        wandb.init(
            project="alphazero-chess",
            config=args,  # Pass argparse values to wandb for sweeps
            mode="disabled" if args.no_wandb else "online"
        )
    
    # The final config is a merge of sweep params and argparse defaults
    config = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AlphaZeroNet().to(device)
    encoder = MoveEncoder()

    full_dataset = load_training_window(encoder)
    if not full_dataset:
        print("No data found, exiting run.", flush=True)

        if wandb.run: 
            wandb.run.finish()
        return

    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    
    train(net, device, train_loader, val_loader, config)

    # Finish the wandb run
    if wandb.run:
        wandb.run.finish()

if __name__ == "__main__":
    main()