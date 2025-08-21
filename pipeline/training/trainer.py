"""Training loop and model training logic."""

import os
import time

from alphazero.utils import format_time
from config import (
    CANDIDATE_MODEL_PATH,
    DEFAULT_POLICY_LOSS_WEIGHT,
    DEFAULT_LOAD_BALANCE_LOSS_WEIGHT,
    DEFAULT_VALUE_LOSS_WEIGHT,
    DEFAULT_LEGALITY_LOSS_WEIGHT,
)
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb


class Trainer:
    """Handles the training and validation of the AlphaZero model."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: dict,
    ):
        """Initialize the trainer.

        Args:
            model: The neural network to train
            device: Device to train on ('cpu' or 'cuda')
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary with training hyperparameters
        """
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_cuda = self.device.type == "cuda"
        self.current_epoch = 0

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.max_epochs)

        self.best_val_loss = float("inf")
        self.best_unweighted_val_loss = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        (
            total_policy_loss,
            total_load_balance_loss,
            total_value_loss,
            total_legality_loss,
        ) = 0.0, 0.0, 0.0, 0.0
        total_weighted_loss = 0.0
        total_unweighted_loss = 0.0
        num_batches = 0
        total_samples = 0
        start_time = time.time()

        try:
            for batch_idx, batch in enumerate(self.train_loader):
                x, pi, z, legality_target = batch
                batch_size = x.size(0)
                total_samples += batch_size

                # Move data to device
                x = x.to(self.device, non_blocking=True)
                pi = pi.to(self.device, non_blocking=True)
                z = z.to(self.device, non_blocking=True)
                legality_target = legality_target.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.is_cuda,
                ):
                    # Forward pass
                    (
                        agg_policy_logits,
                        value_pred,
                        legality_logits,
                        load_balancing_loss,
                    ) = self.model(x)

                    # Calculate losses
                    policy_loss = F.cross_entropy(agg_policy_logits, pi)
                    value_loss = F.mse_loss(value_pred.squeeze(), z.squeeze())
                    legality_loss = F.binary_cross_entropy_with_logits(
                        legality_logits, legality_target
                    )

                    # Calculate unweighted and weighted losses
                    unweighted_batch_loss = (
                        policy_loss + value_loss + legality_loss + load_balancing_loss
                    )
                    batch_loss = (
                        (policy_loss * self.config.get("policy_loss_weight", DEFAULT_POLICY_LOSS_WEIGHT))
                        + (
                            load_balancing_loss
                            * self.config.get("load_balance_loss_weight", DEFAULT_LOAD_BALANCE_LOSS_WEIGHT)
                        )
                        + (value_loss * self.config.get("value_loss_weight", DEFAULT_VALUE_LOSS_WEIGHT))
                        + (legality_loss * self.config.get("legality_loss_weight", DEFAULT_LEGALITY_LOSS_WEIGHT))
                    )

                # Backward pass and optimize
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_load_balance_loss += load_balancing_loss.item()
                total_value_loss += value_loss.item()
                total_legality_loss += legality_loss.item()
                total_weighted_loss += batch_loss.item()
                total_unweighted_loss += unweighted_batch_loss.item()
                num_batches += 1

            # Calculate epoch metrics
            avg_weighted_loss = total_weighted_loss / max(1, num_batches)
            avg_unweighted_loss = total_unweighted_loss / max(1, num_batches)
            avg_policy_loss = total_policy_loss / max(1, num_batches)
            avg_value_loss = total_value_loss / max(1, num_batches)
            avg_legality_loss = total_legality_loss / max(1, num_batches)
            avg_load_balance_loss = total_load_balance_loss / max(1, num_batches)
            epoch_time = time.time() - start_time

            # Print epoch summary with loss components
            print("\n" + "=" * 80)
            print(f"Epoch {self.current_epoch} Complete")
            print("-" * 40)
            print(f"{'Batches:':<15} {num_batches}")
            print(f"{'Samples:':<15} {total_samples}")
            print(f"{'Time:':<15} {format_time(epoch_time)}")
            print("-" * 40)
            print(
                f"{'Policy Loss:':<20} {avg_policy_loss:.4f} (x{self.config.get('policy_loss_weight', DEFAULT_POLICY_LOSS_WEIGHT):.2f} = {self.config.get('policy_loss_weight', DEFAULT_POLICY_LOSS_WEIGHT) * avg_policy_loss:.4f})"
            )
            print(
                f"{'Value Loss:':<20} {avg_value_loss:.4f} (x{self.config.get('value_loss_weight', DEFAULT_VALUE_LOSS_WEIGHT):.2f} = {self.config.get('value_loss_weight', DEFAULT_VALUE_LOSS_WEIGHT) * avg_value_loss:.4f})"
            )
            print(
                f"{'Legality Loss:':<20} {avg_legality_loss:.4f} (x{self.config.get('legality_loss_weight', DEFAULT_LEGALITY_LOSS_WEIGHT):.2f} = {self.config.get('legality_loss_weight', DEFAULT_LEGALITY_LOSS_WEIGHT) * avg_legality_loss:.4f}"
            )
            print(
                f"{'Load Balance Loss:':<20} {avg_load_balance_loss:.4f} (x{self.config.get('load_balance_loss_weight', DEFAULT_LOAD_BALANCE_LOSS_WEIGHT):.2f} = {self.config.get('load_balance_loss_weight', DEFAULT_LOAD_BALANCE_LOSS_WEIGHT) * avg_load_balance_loss:.4f}"
            )
            print("-" * 40)
            print(f"{'UNWEIGHTED LOSS:':<20} {avg_unweighted_loss:.4f}")
            print(f"{'WEIGHTED LOSS:':<20} {avg_weighted_loss:.4f}")
            print("=" * 80 + "\n")

            # Return detailed metrics
            return {
                "train/weighted_loss": avg_weighted_loss,
                "train/unweighted_loss": avg_unweighted_loss,
                "train/policy_loss": avg_policy_loss,
                "train/value_loss": avg_value_loss,
                "train/legality_loss": avg_legality_loss,
                "train/load_balance_loss": avg_load_balance_loss,
                "train/epoch": self.current_epoch,
                "train/samples_per_second": total_samples / max(1, epoch_time),
            }

        except Exception as e:
            print(f"Training epoch failed: {e}", flush=True)
            raise

    def validate(self) -> dict:
        """Run validation on the validation set.

        Returns:
            dict: Dictionary containing validation metrics including:
                - val/weighted_loss: Weighted average of all loss components
                - val/policy_loss: Average policy loss
                - val/value_loss: Average value loss
                - val/legality_loss: Average legality loss
                - val/load_balance_loss: Average load balancing loss
                - val/epoch: Current epoch number
                - val/samples_per_second: Processing speed in samples/second
        """
        self.model.eval()
        (
            total_policy_loss,
            total_load_balance_loss,
            total_value_loss,
            total_legality_loss,
        ) = 0.0, 0.0, 0.0, 0.0
        total_weighted_loss = 0.0
        total_unweighted_loss = 0.0
        num_batches = 0
        total_samples = 0
        start_time = time.time()

        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    x, pi, z, legality_target = batch
                    batch_size = x.size(0)
                    total_samples += batch_size

                    x, pi, z, legality_target = (
                        x.to(self.device, non_blocking=True),
                        pi.to(self.device, non_blocking=True),
                        z.to(self.device, non_blocking=True),
                        legality_target.to(self.device, non_blocking=True),
                    )

                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.is_cuda,
                    ):
                        (
                            agg_policy_logits,
                            value_pred,
                            legality_logits,
                            load_balancing_loss,
                        ) = self.model(x)

                        policy_loss = F.cross_entropy(agg_policy_logits, pi)
                        value_loss = F.mse_loss(value_pred.squeeze(), z.squeeze())
                        legality_loss = F.binary_cross_entropy_with_logits(
                            legality_logits, legality_target
                        )

                        # Calculate unweighted and weighted losses
                        unweighted_batch_loss = (
                            policy_loss
                            + value_loss
                            + legality_loss
                            + load_balancing_loss
                        )
                        weighted_loss = (
                            (policy_loss * self.config.get("policy_loss_weight", DEFAULT_POLICY_LOSS_WEIGHT))
                            + (value_loss * self.config.get("value_loss_weight", DEFAULT_VALUE_LOSS_WEIGHT))
                            + (
                                legality_loss * self.config.get("legality_loss_weight", DEFAULT_LEGALITY_LOSS_WEIGHT)
                            )
                            + (
                                load_balancing_loss * self.config.get("load_balance_loss_weight", DEFAULT_LOAD_BALANCE_LOSS_WEIGHT)
                            )
                        )

                        # Accumulate losses
                        total_weighted_loss += weighted_loss.item()
                        total_unweighted_loss += unweighted_batch_loss.item()
                        total_policy_loss += policy_loss.item()
                        total_load_balance_loss += load_balancing_loss.item()
                        total_value_loss += value_loss.item()
                        total_legality_loss += legality_loss.item()
                        num_batches += 1

            # Calculate average losses
            num_batches = max(1, num_batches)  # Avoid division by zero
            avg_weighted_loss = total_weighted_loss / num_batches
            avg_unweighted_loss = total_unweighted_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_legality_loss = total_legality_loss / num_batches
            avg_load_balance_loss = total_load_balance_loss / num_batches

            val_time = time.time() - start_time
            samples_per_second = total_samples / max(1, val_time)

            # Print validation summary with loss components
            print("\n" + "=" * 80)
            print(f"Validation Results - Epoch {self.current_epoch}")
            print("-" * 40)
            print(f"{'Batches:':<15} {num_batches}")
            print(f"{'Samples:':<15} {total_samples}")
            print(f"{'Time:':<15} {format_time(val_time)}")
            print("-" * 40)
            print(
                f"{'Policy Loss:':<20} {avg_policy_loss:.4f} (x{self.config.get('policy_loss_weight', DEFAULT_POLICY_LOSS_WEIGHT):.2f} = {self.config.get('policy_loss_weight', DEFAULT_POLICY_LOSS_WEIGHT) * avg_policy_loss:.4f})"
            )
            print(
                f"{'Value Loss:':<20} {avg_value_loss:.4f} (x{self.config.get('value_loss_weight', DEFAULT_VALUE_LOSS_WEIGHT):.2f} = {self.config.get('value_loss_weight', DEFAULT_VALUE_LOSS_WEIGHT) * avg_value_loss:.4f})"
            )
            print(
                f"{'Legality Loss:':<20} {avg_legality_loss:.4f} (x{self.config.get('legality_loss_weight', DEFAULT_LEGALITY_LOSS_WEIGHT):.2f} = {self.config.get('legality_loss_weight', DEFAULT_LEGALITY_LOSS_WEIGHT) * avg_legality_loss:.4f}"
            )
            print(
                f"{'Load Balance Loss:':<20} {avg_load_balance_loss:.4f} (x{self.config.get('load_balance_loss_weight', DEFAULT_LOAD_BALANCE_LOSS_WEIGHT):.2f} = {self.config.get('load_balance_loss_weight', DEFAULT_LOAD_BALANCE_LOSS_WEIGHT) * avg_load_balance_loss:.4f}"
            )
            print("-" * 40)
            print(f"{'UNWEIGHTED VAL LOSS:':<20} {avg_unweighted_loss:.4f}")
            print(f"{'WEIGHTED VAL LOSS:':<20} {avg_weighted_loss:.4f}")
            print("=" * 80 + "\n")

            # Return detailed metrics with consistent naming
            return {
                "val/weighted_loss": avg_weighted_loss,
                "val/unweighted_loss": avg_unweighted_loss,
                "val/policy_loss": avg_policy_loss,
                "val/value_loss": avg_value_loss,
                "val/legality_loss": avg_legality_loss,
                "val/load_balance_loss": avg_load_balance_loss,
                "val/epoch": self.current_epoch,
                "val/samples_per_second": samples_per_second,
            }

        except Exception as e:
            print(f"Validation failed: {e}", flush=True)
            raise

    def train(self) -> None:
        """Run the full training loop."""
        total_start_time = time.time()

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                self.current_epoch = epoch
                print(f"\nEpoch {epoch:03d}/{self.config.max_epochs:03d}", flush=True)

                # Training phase
                train_metrics = self.train_epoch()

                # Validation phase
                val_metrics = self.validate()
                checkpoint_loss = val_metrics["val/weighted_loss"]

                # Learning rate scheduling
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Combine metrics for logging
                log_metrics = {
                    **train_metrics,
                    **val_metrics,
                    "epoch": epoch,
                    "lr": current_lr,
                }

                # Log metrics to console
                print(
                    f"  Train Loss: {train_metrics['train/weighted_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val/weighted_loss']:.4f} | "
                    f"LR: {current_lr:.1e}",
                    flush=True,
                )

                # Log to wandb
                if wandb.run and not wandb.run.disabled:
                    wandb.log(log_metrics)

                # Checkpointing
                if checkpoint_loss < self.best_val_loss:
                    self.best_val_loss = checkpoint_loss
                    self.best_unweighted_val_loss = val_metrics["val/unweighted_loss"]
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0

                    # Save the model
                    torch.save(self.model.state_dict(), CANDIDATE_MODEL_PATH)
                    print("  [BEST] New best model saved!", flush=True)

                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.config.patience:
                        print(f"  [STOP] Early stopping at epoch {epoch}", flush=True)
                        break

        except KeyboardInterrupt:
            print("\nTraining interrupted!", flush=True)
        except Exception as e:
            print(f"\nTraining failed: {e}", flush=True)
            raise
        finally:
            total_time = time.time() - total_start_time
            print(f"\nTraining completed in {format_time(total_time)}", flush=True)

            if self.best_epoch != -1:  # Only log if we have a valid best model
                print(
                    f"Best validation loss: {self.best_val_loss:.4f} (unweighted: {self.best_unweighted_val_loss:.4f}) at epoch {self.best_epoch}",
                    flush=True,
                )

                # Log final metrics to wandb
                if wandb.run and not wandb.run.disabled:
                    # Final log with best metrics
                    wandb.log(
                        {
                            "best/weighted_validation_loss": self.best_val_loss,
                            "best/unweighted_validation_loss": self.best_unweighted_val_loss,
                            "best/epoch": self.best_epoch,
                            "total_training_time": total_time,
                            "total_epochs": self.current_epoch,
                        },
                        commit=False,
                    )

                    # Save the best model as an artifact
                    print("--- Logging best model to wandb artifacts ---", flush=True)
                    artifact = wandb.Artifact(
                        name=f"model-best-{wandb.run.id}",
                        type="model",
                        metadata={
                            "best_epoch": self.best_epoch,
                            "best_weighted_validation_loss": self.best_val_loss,
                            "best_unweighted_validation_loss": self.best_unweighted_val_loss,
                            "total_training_time": total_time,
                            "total_epochs": self.current_epoch,
                        },
                    )

                    # Only add the file if it exists
                    if os.path.exists(CANDIDATE_MODEL_PATH):
                        artifact.add_file(CANDIDATE_MODEL_PATH)
                        wandb.run.log_artifact(artifact)
                        print(
                            "  [WANDB] Best model artifact logged successfully",
                            flush=True,
                        )
                    else:
                        print(
                            "  [WARNING] Model file not found for artifact creation",
                            flush=True,
                        )
            else:
                print("  [WARNING] No best model was saved during training", flush=True)
