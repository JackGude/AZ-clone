#!/usr/bin/env python3
"""
Main training script for the AlphaZero model.

This script handles the training pipeline including data loading, model training,
validation, and checkpointing. It uses PyTorch's IterableDataset for efficient
data loading and processing.
"""

import argparse
import os
import random

from alphazero.model import AlphaZeroNet
from alphazero.utils import ensure_project_root
from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    DEFAULT_LEGALITY_LOSS_WEIGHT,
    DEFAULT_LOAD_BALANCE_LOSS_WEIGHT,
    DEFAULT_POLICY_LOSS_WEIGHT,
    DEFAULT_VALUE_LOSS_WEIGHT,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_TRAINING_WORKERS,
    PATIENCE,
    PROJECT_NAME,
    WEIGHT_DECAY,
)
from pipeline.data import StreamingDataset, load_tensor_files_for_training
from pipeline.training import Trainer
import torch
from torch.utils.data import DataLoader
import wandb


# Helper function to handle boolean arguments from strings
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    """Main entry point for training the AlphaZero model."""
    parser = argparse.ArgumentParser(description="Train the AlphaZero model")
    parser.add_argument(
        "--gen-id",
        type=str,
        default="manual",
        help="Generation ID for this run (used for grouping in wandb)",
    )
    # CORRECTED: This new definition robustly handles boolean flags from wandb sweeps
    parser.add_argument(
        "--load-weights",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Load the best model weights before training. Accepts true/false values.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"Weight decay (default: {WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--policy-loss-weight",
        type=float,
        default=DEFAULT_POLICY_LOSS_WEIGHT,
        help=f"Weight for the policy loss component. (default: {DEFAULT_POLICY_LOSS_WEIGHT})",
    )
    parser.add_argument(
        "--load-balance-loss-weight",
        type=float,
        default=DEFAULT_LOAD_BALANCE_LOSS_WEIGHT,
        help=f"Weight for the load balance loss component. (default: {DEFAULT_LOAD_BALANCE_LOSS_WEIGHT})",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=DEFAULT_VALUE_LOSS_WEIGHT,
        help=f"Weight for the value loss component. (default: {DEFAULT_VALUE_LOSS_WEIGHT})",
    )
    parser.add_argument(
        "--legality-loss-weight",
        type=float,
        default=DEFAULT_LEGALITY_LOSS_WEIGHT,
        help=f"Weight for the legality loss component. (default: {DEFAULT_LEGALITY_LOSS_WEIGHT})",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Maximum number of epochs (default: {MAX_EPOCHS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Number of epochs to wait before early stopping (default: {PATIENCE})",
    )
    parser.add_argument(
        "--n-res-blocks",
        type=int,
        default=40,
        help=f"Number of residual blocks in the model's tower. (default: {40})",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=384,
        help=f"Number of channels in the model's convolutional layers. (default: {384})",
    )
    args = parser.parse_args()

    # --- SIMPLIFIED W&B INITIALIZATION ---
    # This is the single source of truth.
    # We pass the 'args' object directly to the config.
    # W&B automatically handles sweeps this way.
    run = wandb.init(
        project=PROJECT_NAME,
        config=args,  # Use the argparse object directly
    )

    # For the rest of the script, wandb.config will be our reliable config object.
    # W&B ensures that its keys use underscores, matching Python attribute access.
    config = wandb.config
    # --- END SIMPLIFICATION ---

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}", flush=True)

        # Initialize model using attribute-style access (e.g., config.n_res_blocks)
        # This now works because 'config' is created from 'args' and has the correct keys.
        net = AlphaZeroNet(
            channels=config.channels,
            n_res_blocks=config.n_res_blocks,
        ).to(device)

        # if device == "cuda":
        #     net = torch.compile(net)

        # Load weights if requested
        if config.load_weights and os.path.exists(BEST_MODEL_PATH):
            print(f"Loading weights from {BEST_MODEL_PATH}...", flush=True)
            net.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
        else:
            print("No weights loaded. Starting from scratch.", flush=True)

        # Get all chunk files
        all_chunk_files = load_tensor_files_for_training()
        if not all_chunk_files:
            print("No data found, exiting run.", flush=True)
            return

        # Split into training and validation sets (80/20 split)
        random.shuffle(all_chunk_files)
        split_idx = int(0.8 * len(all_chunk_files))
        train_files = all_chunk_files[:split_idx]
        val_files = all_chunk_files[split_idx:]

        print(f"Found {len(all_chunk_files)} chunk files", flush=True)
        print(f"Training on {len(train_files)} chunks", flush=True)
        print(f"Validating on {len(val_files)} chunks", flush=True)

        # Create training and validation datasets
        train_dataset = StreamingDataset(train_files, shuffle=True)
        val_dataset = StreamingDataset(val_files, shuffle=False)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=NUM_TRAINING_WORKERS // 2,  # Use integer division
            pin_memory=True,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=NUM_TRAINING_WORKERS // 2,  # Use integer division
            pin_memory=True,
            persistent_workers=True,
        )

        # Initialize and run trainer
        trainer = Trainer(
            model=net,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )

        # Start training
        trainer.train()

    except Exception as e:
        print(f"Error during training: {e}", flush=True)
        if run:
            run.finish(exit_code=1)
        raise

    if run:
        run.finish()


if __name__ == "__main__":
    ensure_project_root()
    main()
