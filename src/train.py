"""Training script for GPT-2 model.

Features:
- Mixed precision training (fp16)
- Gradient accumulation
- Learning rate scheduling with warmup
- Early stopping
- Validation monitoring
- Checkpoint saving
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict
import yaml
from tqdm import tqdm

# Ensure project paths are available (works from root or src/)
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from model.gpt2 import GPT2, GPT2Config

# Dashboard integration (optional - non-critical)
try:
    from dashboard.integration import log_training_step
    DASHBOARD_AVAILABLE = True
    print("[Dashboard] Integration loaded successfully")
except Exception as e:
    DASHBOARD_AVAILABLE = False
    log_training_step = None
    print(f"[Dashboard] Integration not available: {e}")


def compute_gradient_stats(model):
    """Compute gradient statistics for dashboard monitoring."""
    total_norm = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            count += 1
    total_norm = total_norm ** (1. / 2.)
    return total_norm if count > 0 else 0.0


class TokenDataset(Dataset):
    """Dataset for tokenized sequences."""

    def __init__(self, data_path: str):
        """
        Initialize dataset.

        Args:
            data_path: Path to .pt file containing tokenized data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = torch.load(data_path, weights_only=True)
        print(f"Loaded dataset from {data_path}")
        print(f"Shape: {self.data.shape}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get item by index.

        Returns:
            Tensor of token IDs (seq_length,)
        """
        return self.data[idx]


class Trainer:
    """GPT-2 Trainer with modern optimizations."""

    def __init__(self, config: Dict):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = config["system"]["device"]

        # Create directories
        os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["logging"]["log_dir"], exist_ok=True)

        # Initialize model
        print("Initializing model...")
        model_config = GPT2Config(**config["model"])
        self.model = GPT2(model_config).to(self.device)

        # Gradient checkpointing (saves VRAM at cost of ~20% speed)
        if config["training"].get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (VRAM saving mode)")

        # Compile model if enabled (PyTorch 2.0+)
        if "compile_model" not in config["system"]:
            config["system"]["compile_model"] = False

        if config["system"].get("compile_model", False):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Initialize datasets
        print("\nLoading datasets...")
        self.train_dataset = TokenDataset(config["data"]["train_data"])
        self.val_dataset = TokenDataset(config["data"]["val_data"])

        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"].get("num_workers", 0),
            pin_memory=config["data"].get("pin_memory", False),
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 0),
            pin_memory=config["data"].get("pin_memory", False),
        )

        print(f"Config training: {config['training']}")
        print(f"LR: {config['training']['learning_rate']} (type: {type(config['training']['learning_rate'])})")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Calculate total training steps
        steps_per_epoch = len(self.train_loader) // config["training"]["gradient_accumulation_steps"]
        total_steps = steps_per_epoch * config["training"]["num_epochs"]

        # Calculate warmup steps dynamically from ratio (7% is standard)
        warmup_ratio = config["training"].get("warmup_ratio", 0.07)
        warmup_steps = int(total_steps * warmup_ratio)

        # Store for logging
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.steps_per_epoch = steps_per_epoch

        if config["training"]["lr_scheduler"] == "cosine":
            from torch.optim.lr_scheduler import LambdaLR

            # Warmup + Cosine schedule
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        else:  # linear
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision scaler
        self.use_fp16 = config["training"].get("use_fp16", False)
        self.scaler = GradScaler(device='cuda') if self.use_fp16 else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        # Gradient accumulation
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.pad_token_id = config["tokenizer"]["pad_token_id"]

        print(f"\n" + "=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Device:                         {self.device}")
        print(f"Model parameters:               {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Per-device batch size:          {config['training']['batch_size']}")
        print(f"Gradient accumulation steps:    {self.gradient_accumulation_steps}")
        print(f"Effective batch size:           {config['training']['batch_size'] * self.gradient_accumulation_steps}")
        print(f"Mixed precision (FP16):         {self.use_fp16}")
        print()
        print(f"Training samples:               {len(self.train_dataset):,}")
        print(f"Validation samples:             {len(self.val_dataset):,}")
        print(f"Batches per epoch:              {len(self.train_loader):,}")
        print(f"Steps per epoch:                {steps_per_epoch:,}")
        print(f"Number of epochs:               {config['training']['num_epochs']}")
        print(f"Total training steps:           {total_steps:,}")
        print()
        print(f"Learning rate:                  {config['training']['learning_rate']:.2e}")
        print(f"LR scheduler:                   {config['training']['lr_scheduler']}")
        print(f"Warmup ratio:                   {warmup_ratio:.1%}")
        print(f"Warmup steps:                   {warmup_steps:,} ({warmup_steps/total_steps:.1%} of total)")
        print(f"Weight decay:                   {config['training']['weight_decay']}")
        print(f"Gradient clipping:              {config['training']['max_grad_norm']}")
        print(f"Pad token ID:                   {self.pad_token_id}")
        print()
        print(f"Eval interval:                  Every {config['training']['eval_interval']} steps")
        print(f"Save interval (epochs):         Every {config['training'].get('save_interval_epochs', 'N/A')} epochs")
        print(f"Early stopping patience:        {config['training']['early_stopping_patience']}")
        print("=" * 80)

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar for epoch
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config['training']['num_epochs']}",
            total=len(self.train_loader),
            unit="step",
            leave=True,
            ncols=120,
        )

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = batch.to(self.device)

            # Create attention mask to ignore padding tokens
            attention_mask = (batch != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask.float()) * -1e9

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.use_fp16):
                output = self.model(batch, labels=batch, attention_mask=attention_mask)
                logits, loss, hidden_states = output[0], output[1], output[2]

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["max_grad_norm"],
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["max_grad_norm"],
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Track loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Calculate metrics
            avg_loss = total_loss / num_batches
            perplexity = math.exp(min(avg_loss, 10))  # Cap at 10 to avoid overflow
            current_lr = self.scheduler.get_last_lr()[0]

            # Update progress bar with metrics
            postfix_dict = {
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{perplexity:.2f}",
                "lr": f"{current_lr:.2e}",
                "step": f"{self.global_step}/{self.total_steps}",
            }

            # Add GPU memory if CUDA available
            if torch.cuda.is_available():
                postfix_dict["gpu"] = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"

            pbar.set_postfix(postfix_dict)

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            total=len(self.val_loader),
            unit="batch",
            leave=False,
            ncols=100,
        )

        for batch in pbar:
            batch = batch.to(self.device)

            # Create attention mask to ignore padding tokens
            attention_mask = (batch != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask.float()) * -1e9

            with autocast(device_type='cuda', enabled=self.use_fp16):
                output = self.model(batch, labels=batch, attention_mask=attention_mask)
                logits, loss, hidden_states = output[0], output[1], output[2]

            total_loss += loss.item()
            num_batches += 1

            # Calculate and display running metrics
            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{math.exp(min(avg_loss, 10)):.2f}"
            })

        return total_loss / num_batches

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number (1-based)
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        save_interval_epochs = self.config["training"].get("save_interval_epochs")

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            self.model.save_pretrained(str(best_path))
            # No print here - we print in the train() method with more context

        # Save intermediate checkpoint
        if save_interval_epochs and epoch % save_interval_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            self.model.save_pretrained(str(checkpoint_path))
            # print(f"✓ Saved intermediate checkpoint: {checkpoint_path.name}")


    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

        num_epochs = self.config["training"]["num_epochs"]
        patience = self.config["training"]["early_stopping_patience"]
        early_stopped = False

        epochs_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch", colour="green")

        for epoch in epochs_pbar:
            self.epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate at end of epoch
            val_loss = self.validate()
            perplexity = math.exp(val_loss)
            epoch_time = time.time() - epoch_start

            # Early stopping check at end of epoch
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{num_epochs} SUMMARY")
            print(f"{'='*80}")
            print(f"Train Loss:      {train_loss:.4f}")
            print(f"Val Loss:        {val_loss:.4f}")
            print(f"Perplexity:      {perplexity:.2f}")
            print(f"Epoch Time:      {epoch_time:.1f}s")
            print(f"Best Val Loss:   {self.best_val_loss:.4f} (Epoch {self.best_epoch})")

            # Log to dashboard every epoch
            if DASHBOARD_AVAILABLE and log_training_step:
                try:
                    current_lr = self.scheduler.get_last_lr()[0]
                    gradient_norm = compute_gradient_stats(self.model)
                    log_training_step(
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        perplexity=perplexity,
                        learning_rate=current_lr,
                        epoch_time=epoch_time,
                        gradient_norm=gradient_norm
                    )
                except Exception as e:
                    print(f"[Dashboard] Logging error (non-critical): {e}")

                # Update training data projection every epoch for real-time dashboard updates
                try:
                    from dashboard.training_hooks import save_training_data_projection
                    save_training_data_projection(self.model, train_data_path="data/train.pt", sample_size=1000)
                except Exception as e:
                    print(f"[Dashboard] Training data projection update skipped (non-critical): {e}")

            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                # self.save_checkpoint(is_best=True)
                self.save_checkpoint(epoch=epoch + 1, is_best=True)
                print(f"\n✓ Validation loss improved by {improvement:.4f}! Saving checkpoint.")
                print(f"  Patience reset: 0/{patience}")
            else:
                self.save_checkpoint(epoch=epoch + 1, is_best=False)
                self.patience_counter += 1
                print(f"\n✗ No improvement in validation loss.")
                print(f"  Patience: {self.patience_counter}/{patience}")

                if self.patience_counter >= patience:
                    print(f"\n{'='*80}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"{'='*80}")
                    print(f"Best validation loss not improved for {patience} consecutive epochs.")
                    print(f"Stopping training at epoch {epoch + 1}/{num_epochs}.")
                    print(f"Best model from epoch {self.best_epoch} with val_loss={self.best_val_loss:.4f}")
                    print(f"{'='*80}\n")
                    early_stopped = True
                    break

            print(f"{'='*80}\n")

            # Update progress bar
            epochs_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'perplexity': f'{perplexity:.2f}',
                'time': f'{epoch_time:.1f}s',
                'patience': f'{self.patience_counter}/{patience}'
            })

        # Save final model
        final_path = Path(self.config["logging"]["checkpoint_dir"]) / "final_model.pt"
        self.model.save_pretrained(str(final_path))

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        if early_stopped:
            print(f"Training stopped early at epoch {self.epoch + 1}/{num_epochs}")
        else:
            print(f"Completed all {num_epochs} epochs")
        print(f"Best model: Epoch {self.best_epoch} with val_loss={self.best_val_loss:.4f}")
        print(f"Best model saved at: {Path(self.config['logging']['checkpoint_dir']) / 'best_model.pt'}")
        print(f"Final model saved at: {final_path}")
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    # Load config (works from root or src/)
    config_path = PROJECT_ROOT / "config" / "gpt2_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set device
    if config["system"]["device"] == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["system"]["device"] = "cpu"

    # Set random seed
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize trainer
    trainer = Trainer(config)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()