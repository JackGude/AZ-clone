# train.py

import os
import argparse
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import time
import json

from config import (
    # Project and File Paths
    PROJECT_NAME,
    TENSOR_CACHE_DIR,
    BEST_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    # Training Config
    NUM_TRAINING_WORKERS,
    WARMUP_LEARNING_RATE,
    WARMUP_WEIGHT_DECAY,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    TRAIN_WINDOW_SIZE,
)
from alphazero.model import AlphaZeroNet
from alphazero.utils import format_time

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset Class & Data Loading
# ─────────────────────────────────────────────────────────────────────────────


class TensorDataset(Dataset):
    """
    A very simple PyTorch Dataset that loads ready-to-use tensors from a list of file paths.
    """
    def __init__(self, file_paths: list):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load a pre-processed tensor tuple (state, pi, z)
        return torch.load(self.file_paths[idx], weights_only=True)


def load_tensor_files_for_training():
    """
    Loads a subset of the most recent .pt files from the replay cache for training.
    """
    
    # print(f"Searching for training data files in '{TENSOR_CACHE_DIR}'...", flush=True)
    if not os.path.isdir(TENSOR_CACHE_DIR):
        print(f"Error: Replay cache directory '{TENSOR_CACHE_DIR}' not found.", flush=True)
        return None

    tensor_files = [os.path.join(TENSOR_CACHE_DIR, f) for f in os.listdir(TENSOR_CACHE_DIR) if f.endswith(".pt")]

    if not tensor_files:
        print(f"No training data files found in '{TENSOR_CACHE_DIR}'. Cannot train.", flush=True)
        return None

    # Use the most recent files up to the window size
    tensor_files.sort(key=os.path.getmtime, reverse=True)
    
    # Select a sample from the most recent files
    files_to_load = tensor_files[:TRAIN_WINDOW_SIZE]
    
    # If the buffer is smaller than the window, use all files
    if len(tensor_files) < TRAIN_WINDOW_SIZE:
        print(f"Found {len(tensor_files)} total positions, which is less than the training window size of {TRAIN_WINDOW_SIZE}. Using all positions.", flush=True)
    else:
        print(f"Found {len(tensor_files)} total positions. Loading a window of the most recent {TRAIN_WINDOW_SIZE}.", flush=True)
        
    return TensorDataset(files_to_load)


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
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
            f"Epoch {epoch:02d}/{config.max_epochs} | Validation Loss: {avg_val_loss:.4f} | Training Loss: {avg_train_loss:.4f} | Learning Rate: {current_lr:.1e}",
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
            print("  Validation loss improved. Saved new candidate model.", flush=True)
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
        "early_stopped": epochs_without_improvement >= config.patience,
    }

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    # --- Artifact Logging ---
    if best_epoch != -1 and wandb.run and not wandb.run.disabled:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Block
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """
    This script's main function is designed for three execution contexts:
    1. Pipeline Mode: Called by `automate.py` with specific arguments.
    2. Manual Mode: Run directly by a user with default arguments.
    3. Sweep Mode: Called by `wandb agent` for hyperparameter sweeps.
    """
    parser = argparse.ArgumentParser(description="Train a new AlphaZero model.")
    # --- Hyperparameters (for sweeps and manual runs) ---
    parser.add_argument(
        "--learning-rate", type=float, default=WARMUP_LEARNING_RATE, help="Learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WARMUP_WEIGHT_DECAY, help="Weight decay."
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Maximum number of epochs to train for. Defaults to {MAX_EPOCHS}.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Number of epochs to wait before early stopping. Defaults to {PATIENCE}.",
    )
    # --- Control Arguments ---
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging."
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load weights from best.pth to continue training. Defaults to False.",
    )
    # --- Arguments for Pipeline/Manual mode ---
    parser.add_argument(
        "--gen-id", type=str, default="manual", help="Generation ID for this run."
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default="logs/training_results.json",
        help="Path to write training results.\nDefault: logs/training_results.json",
    )
    args = parser.parse_args()

    # --- Initialize wandb based on the execution context ---
    is_sweep = os.getenv("WANDB_SWEEP_ID") is not None

    if is_sweep:
        # --- Sweep Mode ---
        # W&B agent will pass all hyperparameters in `config`.
        # No need to specify group, name, or job_type as the sweep controls this.
        run = wandb.init(
            project=PROJECT_NAME,
            config=vars(args),
            mode="disabled" if args.no_wandb else "online",
            job_type="sweep",
        )

    else:
        # --- Single Run Mode (Pipeline or Manual) ---
        run = wandb.init(
            project=PROJECT_NAME,
            config=vars(args),
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
            print("No weights loaded. Starting from scratch.", flush=True)

        full_dataset = load_tensor_files_for_training()
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
            num_workers=NUM_TRAINING_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=NUM_TRAINING_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )

        train(net, device, train_loader, val_loader, config, args.result_file)

    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
