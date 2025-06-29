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

import torch
import wandb
from torch.utils.data import DataLoader

from alphazero.model import AlphaZeroNet
from alphazero.utils import ensure_project_root
from pipeline.data import load_tensor_files_for_training, StreamingDataset
from pipeline.training import Trainer

from config import (
    NUM_TRAINING_WORKERS,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    PROJECT_NAME,
    WEIGHT_DECAY,
)


def main():
    """Main entry point for training the AlphaZero model."""
    parser = argparse.ArgumentParser(description="Train the AlphaZero model")
    parser.add_argument(
        "--gen-id",
        type=str,
        default="manual",
        help="Generation ID for this run (used for grouping in wandb)",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load the best model weights before training",
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
    args = parser.parse_args()

    # Check if this is a W&B sweep run
    is_sweep = bool(os.environ.get("WANDB_SWEEP_ID"))

    # Initialize wandb
    if not is_sweep:
        run = wandb.init(
            project=PROJECT_NAME,
            group=args.gen_id,
            name=f"{args.gen_id}-training",
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "gen_id": args.gen_id,
            },
        )
        config = wandb.config
    else:
        run = wandb.init()
        config = wandb.config

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}", flush=True)
        
        # Initialize model
        net = AlphaZeroNet().to(device)

        # Load weights if requested
        if args.load_weights and os.path.exists(BEST_MODEL_PATH):
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
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=NUM_TRAINING_WORKERS // 2,  # Use integer division
            pin_memory=True,
            persistent_workers=True
        )

        # Initialize and run trainer
        trainer = Trainer(
            model=net,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
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
