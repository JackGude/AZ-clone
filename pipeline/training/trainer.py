"""Training loop and model training logic."""

import time

from alphazero.utils import format_time
from config import CANDIDATE_MODEL_PATH
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

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.max_epochs)

        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss, num_batches = 0.0, 0
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
                    agg_policy_logits, value_pred, legality_logits = self.model(x)

                    # Calculate losses
                    policy_loss = F.cross_entropy(agg_policy_logits, pi)
                    value_loss = F.mse_loss(value_pred.squeeze(), z.squeeze())
                    legality_loss = F.binary_cross_entropy_with_logits(
                        legality_logits, legality_target
                    )

                    # Combined loss
                    batch_loss = policy_loss + value_loss + (0.5 * legality_loss)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # Track metrics
                total_loss += batch_loss.item()
                num_batches += 1

            # Calculate epoch metrics
            avg_loss = total_loss / max(1, num_batches)
            epoch_time = time.time() - start_time

            print(
                f"Epoch complete | "
                f"Batches: {num_batches} | "
                f"Samples: {total_samples} | "
                f"Time: {epoch_time:.1f}s | "
                f"Avg Loss: {avg_loss:.4f}",
                flush=True,
            )

            return avg_loss

        except Exception as e:
            print(f"Training epoch failed: {e}", flush=True)
            raise

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self) -> float:
        """Run validation on the validation set."""
        self.model.eval()
        total_loss, num_batches = 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, pi, z, legality_target = batch
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
                    agg_policy_logits, value_pred, legality_logits = self.model(x)

                    policy_loss = F.cross_entropy(agg_policy_logits, pi)
                    value_loss = F.mse_loss(value_pred.squeeze(), z.squeeze())
                    legality_loss = F.binary_cross_entropy_with_logits(
                        legality_logits, legality_target
                    )

                    loss = policy_loss + value_loss + (0.5 * legality_loss)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self) -> None:
        """Run the full training loop."""
        total_start_time = time.time()

        try:
            for epoch in range(1, self.config.max_epochs + 1):
                print(f"\nEpoch {epoch:03d}/{self.config.max_epochs:03d}", flush=True)

                # Training phase
                train_loss = self.train_epoch()

                # Validation phase
                val_loss = self.validate()

                # Learning rate scheduling
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Log metrics
                print(
                    f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.1e}",
                    flush=True,
                )

                # Log to wandb
                if wandb.run and not wandb.run.disabled:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss": train_loss,
                            "val/loss": val_loss,
                            "lr": current_lr,
                        }
                    )

                # Checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), CANDIDATE_MODEL_PATH)
                    print("  [BEST] New best model saved!", flush=True)

                    if wandb.run and not wandb.run.disabled:
                        artifact = wandb.Artifact(
                            f"model-{wandb.run.id}",
                            type="model",
                            metadata={
                                "epoch": epoch,
                                "val_loss": val_loss,
                                "train_loss": train_loss,
                            },
                        )
                        artifact.add_file(CANDIDATE_MODEL_PATH)
                        wandb.log_artifact(artifact)
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
            print(
                f"Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}",
                flush=True,
            )

        # Log final metrics
        if wandb.run and not wandb.run.disabled:
            wandb.log(
                {
                    "best_validation_loss": self.best_val_loss,
                    "total_training_time": time.time() - total_start_time,
                    "best_epoch": self.best_epoch,
                }
            )

            print("--- Logging best model to wandb artifacts ---", flush=True)
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                metadata={
                    "best_epoch": self.best_epoch,
                    "best_validation_loss": self.best_val_loss,
                },
            )
            artifact.add_file(CANDIDATE_MODEL_PATH)
            wandb.run.log_artifact(artifact)
