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
import json

from config import (
    # Project and File Paths
    PROJECT_NAME,
    REPLAY_DIR,
    BEST_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    # Training Config
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    TRAIN_WINDOW_SIZE
)
from alphazero.model import AlphaZeroNet
from alphazero.move_encoder import MoveEncoder

from automate import format_time

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
    game_files = [
        os.path.join(REPLAY_DIR, f)
        for f in os.listdir(REPLAY_DIR)
        if f.endswith(".pkl")
    ]

    if not game_files:
        print("No game files found. Cannot train.", flush=True)
        return None

    print(
        f"Found {len(game_files)} total games. Loading a random window of ~{TRAIN_WINDOW_SIZE} positions...",
        flush=True,
    )
    all_examples = []
    # Keep loading random games until we hit our window size
    while len(all_examples) < TRAIN_WINDOW_SIZE and game_files:
        file_path = random.choice(game_files)
        game_files.remove(file_path)  # Avoid picking the same game twice
        try:
            with open(file_path, "rb") as f:
                all_examples.extend(pickle.load(f))
        except Exception as e:
            print(
                f"Warning: Could not load or process {file_path}. Error: {e}",
                flush=True,
            )

    if not all_examples:
        print(
            "Failed to load any valid examples from the files sampled. Cannot train.",
            flush=True,
        )
        return None

    print(f"Loaded {len(all_examples)} positions for this training run.", flush=True)
    return AZDataset(all_examples, encoder)


# ─────────────────────────────────────────────────────────────────────────────
#  Training Function
# ─────────────────────────────────────────────────────────────────────────────


def train(net, device, train_loader, val_loader, config, result_file):
    """
    The main training and validation loop for the model.
    It now takes a single 'config' object for all hyperparameters.
    
    Args:
        net: The neural network to train
        device: Device to train on ('cpu' or 'cuda')
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration object containing hyperparameters
        result_file: Path to write the training results JSON file
    """
    is_cuda = device == "cuda"
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    total_start_time = time.time()

    for epoch in range(1, config.max_epochs + 1):
        print(f"Beginning epoch {epoch} of {config.max_epochs}...", flush=True)
        epoch_start_time = time.time()
        # --- Training Phase ---
        net.train()
        total_train_loss, train_batches = 0.0, 0
        for x, pi, z in train_loader:
            x, pi, z = x.to(device), pi.to(device), z.to(device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=is_cuda
            ):
                logits, v = net(x)
                loss = F.cross_entropy(logits, pi) + F.mse_loss(
                    v.squeeze(), z.squeeze()
                )
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
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=is_cuda
                ):
                    logits, v = net(x)
                    loss = F.cross_entropy(logits, pi) + F.mse_loss(
                        v.squeeze(), z.squeeze()
                    )
                total_val_loss += loss.item()
                val_batches += 1
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{config.max_epochs} | Val Loss: {avg_val_loss:.4f} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.1e}",
            flush=True,
        )
        print(f"Time: {format_time(time.time() - epoch_start_time)}", flush=True)
        wandb.log(
            {
                "epoch": epoch,
                "validation_loss": avg_val_loss,
                "training_loss": avg_train_loss,
                "learning_rate": current_lr,
                "epoch_time": time.time() - epoch_start_time,
            }
        )

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(net.state_dict(), CANDIDATE_MODEL_PATH)
            print(f"  Validation loss improved. Saved new candidate model.", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(
                    f"  Stopping early as validation loss has not improved for {config.patience} epochs.",
                    flush=True,
                )
                break

    total_training_time = time.time() - total_start_time
    
    # Write training results to JSON file
    results = {
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_training_time": total_training_time,
        "final_learning_rate": current_lr,
        "early_stopped": epochs_without_improvement >= config.patience
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # --- Artifact Logging ---
    if best_epoch != -1:
        wandb.log(
            {
                "best_validation_loss": best_val_loss,
                "total_training_time": total_training_time,
            }
        )
        print("--- Logging best model to wandb artifacts ---", flush=True)
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata={"best_epoch": best_epoch, "best_validation_loss": best_val_loss},
        )
        artifact.add_file(CANDIDATE_MODEL_PATH)
        wandb.run.log_artifact(artifact)


# --- Main Execution Block ---
def main():
    """
    This script's main function is designed for three execution contexts:
    1. Pipeline Mode: Called by `automate.py` with specific arguments.
    2. Manual Mode: Run directly by a user with default arguments.
    3. Sweep Mode: Called by `wandb agent` for hyperparameter sweeps.
    """
    parser = argparse.ArgumentParser(description="Train a new AlphaZero model.")
    # --- Hyperparameters (for sweeps and manual runs) ---
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS, help="Maximum number of epochs to train for.")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Number of epochs to wait before early stopping.")
    # --- Control Arguments ---
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--load-weights", action="store_true", help="Load weights from best.pth to continue training.")
    # --- Arguments for Pipeline/Manual mode ---
    parser.add_argument("--gen-id", type=str, default="manual", help="Generation ID for this run.")
    parser.add_argument("--result-file", type=str, default="training_results.json", help="Path to write training results.")
    args = parser.parse_args()

    # --- Initialize wandb based on the execution context ---
    is_sweep = os.getenv("WANDB_SWEEP_ID") is not None

    if is_sweep:
        # --- Sweep Mode ---
        # W&B agent will pass all hyperparameters in `config`.
        # No need to specify group, name, or job_type as the sweep controls this.
        run = wandb.init(
            project=PROJECT_NAME,
            config=args,
            mode="disabled" if args.no_wandb else "online",
        )
        # Update job_type after init for clarity in the W&B dashboard
        wandb.run.job_type = "sweep"

    else:
        # --- Single Run Mode (Pipeline or Manual) ---
        run = wandb.init(
            project=PROJECT_NAME,
            config=args,
            group=args.gen_id,
            name=f"{args.gen_id}-training",
            job_type="training",
            mode="disabled" if args.no_wandb else "online",
        )

    try:
        config = wandb.config       
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = AlphaZeroNet().to(device)

        if args.load_weights:
            print(f"Loading weights from {BEST_MODEL_PATH}...", flush=True)
            net.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
        else:
            print(f"No weights loaded. Starting from scratch.", flush=True)

        encoder = MoveEncoder()

        full_dataset = load_training_window(encoder)
        if not full_dataset:
            print("No data found, exiting run.", flush=True)
            return

        n_val = int(0.2 * len(full_dataset))
        n_train = len(full_dataset) - n_val

        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

        train(net, device, train_loader, val_loader, config, args.result_file)

    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
