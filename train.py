# train.py

import os
import pickle
import json
import time
import random
import numpy as np
import argparse
import wandb
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from alphazero.model import AlphaZeroNet # Imports the model with SE-Blocks
from alphazero.move_encoder import MoveEncoder
from alphazero.state_encoder import encode_history

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
    print(f"Searching for game files in '{REPLAY_DIR}'...")
    game_files = [os.path.join(REPLAY_DIR, f) for f in os.listdir(REPLAY_DIR) if f.endswith('.pkl')]
    
    if not game_files:
        print("No game files found. Cannot train.")
        return None

    print(f"Found {len(game_files)} total games. Loading a random window of ~{TRAIN_WINDOW_SIZE} positions...")
    all_examples = []
    # Keep loading random games until we hit our window size
    while len(all_examples) < TRAIN_WINDOW_SIZE and game_files:
        file_path = random.choice(game_files)
        game_files.remove(file_path) # Avoid picking the same game twice
        try:
            with open(file_path, "rb") as f:
                all_examples.extend(pickle.load(f))
        except Exception as e:
            print(f"Warning: Could not load or process {file_path}. Error: {e}")

    if not all_examples:
        print("Failed to load any valid examples from the files sampled. Cannot train.")
        return None
        
    print(f"Loaded {len(all_examples)} positions for this training run.")
    return AZDataset(all_examples, encoder)

# ─────────────────────────────────────────────────────────────────────────────
#  Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train(net, device, train_loader, val_loader, args):
    """
    The main training and validation loop for the model.

    Args:
        net (nn.Module): The model to be trained.
        device (str): The device to train on ('cpu' or 'cuda').
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        args (Namespace): Parsed command-line arguments containing hyperparameters.
    """
    is_cuda = (device == "cuda")
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        # --- Training Phase ---
        net.train()
        total_train_loss, train_batches = 0.0, 0
        for x, pi, z in train_loader:
            x, pi, z = x.to(device), pi.to(device), z.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=is_cuda):
                logits, v = net(x)
                loss = F.cross_entropy(logits, pi) + F.mse_loss(v, z)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            train_batches += 1
        avg_train_loss = total_train_loss / train_batches

        # --- Validation Phase ---
        net.eval()
        total_val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for x, pi, z in val_loader:
                x, pi, z = x.to(device), pi.to(device), z.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=is_cuda):
                    logits, v = net(x)
                    loss = F.cross_entropy(logits, pi) + F.mse_loss(v, z)
                total_val_loss += loss.item()
                val_batches += 1
        avg_val_loss = total_val_loss / val_batches
        
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{args.max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {lr:.1e}")

        # Log metrics to wandb if active
        if wandb.run:
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "learning_rate": lr})
        
        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save as candidate.pth for the automation script to pick up
            torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, "candidate.pth"))
            print(f"  Validation loss improved. Saved new candidate model.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  Stopping early as validation loss has not improved for {args.patience} epochs.")
                break

    if best_epoch != -1 and wandb.run:
        print("--- Logging best model to wandb artifacts ---")
        
        # Use the unique run name for the artifact for better tracking
        artifact_name = f"model-{wandb.run.name}"

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"The best model checkpoint from sweep trial {wandb.run.name}.",
            metadata={"best_epoch": best_epoch, "validation_loss": best_val_loss}
        )
        artifact.add_file(os.path.join(CHECKPOINT_DIR, "candidate.pth"))
        wandb.run.log_artifact(artifact)

# ─────────────────────────────────────────────────────────────────────────────
#  Wandb Sweep & Main Execution
# ─────────────────────────────────────────────────────────────────────────────

def sweep_iteration():
    """A single run of a wandb sweep agent."""
    # The agent calls wandb.init() and passes config values
    run = wandb.init()
    args = wandb.config # Get hyperparameters from the sweep controller
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AlphaZeroNet().to(device)

    encoder = MoveEncoder()
    full_dataset = load_training_window(encoder)
    if not full_dataset: return

    # Create data loaders (batch size is also a swept parameter)
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    train(net, device, train_loader, val_loader, args)
    run.finish()

def main(args):
    """Main function for a standalone training run."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize wandb based on --no-wandb flag
    wandb_mode = "disabled" if args.no_wandb else "online"
    # This init is for attaching to an automate.py run or for a single test run.
    wandb.init(project="alphazero-chess", resume="allow", mode=wandb_mode)
    if not args.no_wandb:
        wandb.config.update(args) # Log command-line args to wandb

    # Create model and load data
    net = AlphaZeroNet().to(device)
    encoder = MoveEncoder()
    full_dataset = load_training_window(encoder)
    if not full_dataset: return

    n_val, n_train = int(0.2 * len(full_dataset)), len(full_dataset) - int(0.2 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train(net, device, train_loader, val_loader, args)
    
    if wandb.run:
        wandb.run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlphaZeroNet or run a wandb sweep.")
    # Hyperparameters for a standalone run
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=1536, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    
    # Flags to control execution mode
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--sweep-id", type=str, default=None, help="Wandb sweep ID to start an agent.")
    
    args = parser.parse_args()

    if args.sweep_id:
        # If a sweep ID is provided, start a wandb agent
        print(f"Starting wandb agent for sweep: {args.sweep_id}")
        wandb.agent(args.sweep_id, function=sweep_iteration, project="alphazero-chess", count=10)
    else:
        # Otherwise, run a single training session
        main(args)