"""Multi-phase training for GPT-2 with curriculum and contrastive learning.

Phase 1 — Denoising CLM:
    Corrupt 15% of input tokens, train model to predict clean next tokens.
    Forces robust language understanding.

Phase 2 — Curriculum CLM:
    Standard CLM but data ordered easy → hard (short/simple first, long/complex last).
    Progressive difficulty prevents catastrophic forgetting.

Phase 3 — CLM + Contrastive:
    Standard CLM loss + SimCSE contrastive loss on hidden states.
    Same input, two forward passes with different dropout → positive pair.
    Tightens embedding space: similar texts cluster together.

Usage:
    python src/train_multiphase.py
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict
import yaml
from tqdm import tqdm

# Ensure project paths are available
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
from curriculum import (
    DenoisingCollator,
    ContrastiveLoss,
    create_curriculum_subsets,
    mean_pool,
)

# Dashboard integration (optional)
try:
    from dashboard.integration import log_training_step
    DASHBOARD_AVAILABLE = True
except Exception:
    DASHBOARD_AVAILABLE = False
    log_training_step = None


class TokenDataset(Dataset):
    """Dataset for tokenized sequences."""

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = torch.load(data_path, weights_only=True)
        print(f"Loaded dataset: {data_path} — shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiPhaseTrainer:
    """GPT-2 trainer with 3-phase curriculum."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = config["system"]["device"]

        # Directories
        os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["logging"]["log_dir"], exist_ok=True)

        # Model
        print("Initializing model...")
        model_config = GPT2Config(**config["model"])
        self.model = GPT2(model_config).to(self.device)

        if config["training"].get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Datasets
        print("\nLoading datasets...")
        self.train_dataset = TokenDataset(config["data"]["train_data"])
        self.val_dataset = TokenDataset(config["data"]["val_data"])

        # Tokenizer config
        self.pad_token_id = config["tokenizer"]["pad_token_id"]
        self.vocab_size = config["model"]["vocab_size"]

        # Phase config
        phase_config = config.get("multiphase", {})
        self.phase1_epochs = phase_config.get("phase1_epochs", 3)
        self.phase2_epochs = phase_config.get("phase2_epochs", 10)
        self.phase3_epochs = phase_config.get("phase3_epochs", 7)
        self.corruption_rate = phase_config.get("corruption_rate", 0.15)
        self.contrastive_weight = phase_config.get("contrastive_weight", 0.1)
        self.contrastive_temp = phase_config.get("contrastive_temperature", 0.05)
        self.num_curriculum_buckets = phase_config.get("num_curriculum_buckets", 5)
        self.total_epochs = self.phase1_epochs + self.phase2_epochs + self.phase3_epochs

        # Denoising collator
        self.denoising = DenoisingCollator(
            corruption_rate=self.corruption_rate,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

        # Contrastive loss
        self.contrastive_loss_fn = ContrastiveLoss(temperature=self.contrastive_temp)

        # Curriculum buckets (precomputed)
        print("\nScoring data difficulty for curriculum...")
        self.curriculum_buckets = create_curriculum_subsets(
            self.train_dataset,
            self.train_dataset.data,
            pad_token_id=self.pad_token_id,
            num_buckets=self.num_curriculum_buckets,
        )
        for i, bucket in enumerate(self.curriculum_buckets):
            print(f"  Bucket {i+1}: {len(bucket):,} samples (cumulative)")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # LR scheduler (calculated over total phases)
        self.batch_size = config["training"]["batch_size"]
        self.grad_accum_steps = config["training"]["gradient_accumulation_steps"]
        max_steps_estimate = (len(self.train_dataset) // self.batch_size // self.grad_accum_steps) * self.total_epochs
        warmup_steps = int(max_steps_estimate * config["training"].get("warmup_ratio", 0.07))

        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps_estimate - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.use_fp16 = config["training"].get("use_fp16", False)
        self.scaler = GradScaler(device='cuda') if self.use_fp16 else None

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        # Print config
        mc = config["model"]
        tc = config["training"]
        print(f"\n{'='*80}")
        print("MULTI-PHASE TRAINING CONFIGURATION")
        print(f"{'='*80}")
        print(f"Device:                         {self.device}")
        print(f"Model parameters:               {self.model.get_num_params()/1e6:.2f}M")
        print()
        print(f"--- Architecture ---")
        print(f"n_embd:                         {mc['n_embd']}")
        print(f"n_layer:                         {mc['n_layer']}")
        print(f"n_head:                         {mc['n_head']}")
        print(f"n_inner:                         {mc['n_inner']} (auto-adjusted: {model_config.n_inner})")
        print(f"Vocab size:                     {mc['vocab_size']:,}")
        print(f"Context length:                 {mc['n_positions']}")
        print()
        print(f"--- Modern Features ---")
        print(f"RoPE:                           {mc.get('use_rope', False)}")
        print(f"RMSNorm:                        {mc.get('use_rmsnorm', False)}")
        print(f"SwiGLU:                         {mc.get('use_swiglu', False)}")
        print(f"QK-Norm:                        {mc.get('use_qk_norm', False)}")
        print(f"Flash Attention:                {mc.get('use_flash_attention', False)}")
        print(f"GQA:                            {mc.get('use_gqa', False)}")
        print(f"Label smoothing:                {mc.get('label_smoothing', 0.0)}")
        print()
        print(f"--- Training ---")
        print(f"Per-device batch size:          {self.batch_size}")
        print(f"Gradient accumulation steps:    {self.grad_accum_steps}")
        print(f"Effective batch size:           {self.batch_size * self.grad_accum_steps}")
        print(f"Mixed precision (FP16):         {self.use_fp16}")
        print(f"Gradient checkpointing:         {tc.get('gradient_checkpointing', False)}")
        print(f"Learning rate:                  {tc['learning_rate']:.2e}")
        print(f"LR scheduler:                   {tc['lr_scheduler']}")
        print(f"Warmup ratio:                   {tc.get('warmup_ratio', 0.07):.0%}")
        print(f"Warmup steps:                   {warmup_steps:,}")
        print(f"Weight decay:                   {tc['weight_decay']}")
        print(f"Gradient clipping:              {tc['max_grad_norm']}")
        print(f"Early stopping patience:        {tc['early_stopping_patience']}")
        print()
        print(f"--- Phases ---")
        print(f"Phase 1 — Denoising CLM:        {self.phase1_epochs} epochs (corruption={self.corruption_rate:.0%})")
        print(f"Phase 2 — Curriculum CLM:        {self.phase2_epochs} epochs ({self.num_curriculum_buckets} buckets)")
        print(f"Phase 3 — CLM + Contrastive:     {self.phase3_epochs} epochs (weight={self.contrastive_weight}, temp={self.contrastive_temp})")
        print(f"Total epochs:                   {self.total_epochs}")
        print()
        print(f"--- Data ---")
        print(f"Train samples:                  {len(self.train_dataset):,}")
        print(f"Val samples:                    {len(self.val_dataset):,}")
        print(f"Total steps (estimated):        {max_steps_estimate:,}")
        print(f"{'='*80}\n")

    def _make_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.config["data"].get("num_workers", 0),
            pin_memory=self.config["data"].get("pin_memory", False),
        )

    def _make_attention_mask(self, batch):
        mask = (batch != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        return (1.0 - mask.float()) * -1e9

    def _train_step(self, batch, labels, attention_mask):
        """Single forward + backward step, returns loss."""
        with autocast(device_type='cuda', enabled=self.use_fp16):
            logits, loss, hidden_states, _ = self.model(
                batch, labels=labels, attention_mask=attention_mask,
            )
        return loss, hidden_states

    def _optimizer_step(self, loss):
        """Backward pass + optimizer step with grad accumulation."""
        scaled_loss = loss / self.grad_accum_steps
        if self.use_fp16:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _step_if_ready(self, batch_idx):
        """Execute optimizer step if gradient accumulation is complete."""
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            max_norm = self.config["training"]["max_grad_norm"]
            if self.use_fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

    # =====================================================================
    # Phase 1: Denoising CLM
    # =====================================================================
    def train_phase1_epoch(self) -> float:
        """Train one epoch with denoising corruption."""
        self.model.train()
        loader = self._make_loader(self.train_dataset)
        total_loss, num_batches = 0.0, 0

        pbar = tqdm(loader, desc="  Phase 1 — Denoising", unit="step", ncols=120, leave=True)
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Corrupt input, keep clean labels
            corrupted, labels = self.denoising(batch)
            attention_mask = self._make_attention_mask(corrupted)

            loss, _ = self._train_step(corrupted, labels, attention_mask)
            self._optimizer_step(loss)
            self._step_if_ready(batch_idx)

            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{math.exp(min(avg_loss, 10)):.1f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return total_loss / num_batches

    # =====================================================================
    # Phase 2: Curriculum CLM
    # =====================================================================
    def train_phase2_epoch(self, epoch_in_phase: int) -> float:
        """Train one epoch with curriculum data ordering."""
        self.model.train()

        # Select bucket based on progress (linear schedule)
        progress = epoch_in_phase / max(self.phase2_epochs - 1, 1)
        bucket_idx = min(int(progress * self.num_curriculum_buckets), self.num_curriculum_buckets - 1)
        subset = self.curriculum_buckets[bucket_idx]
        loader = self._make_loader(subset)

        total_loss, num_batches = 0.0, 0
        desc = f"  Phase 2 — Curriculum (bucket {bucket_idx+1}/{self.num_curriculum_buckets}, {len(subset):,} samples)"

        pbar = tqdm(loader, desc=desc, unit="step", ncols=120, leave=True)
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            attention_mask = self._make_attention_mask(batch)

            loss, _ = self._train_step(batch, batch, attention_mask)
            self._optimizer_step(loss)
            self._step_if_ready(batch_idx)

            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{math.exp(min(avg_loss, 10)):.1f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return total_loss / num_batches

    # =====================================================================
    # Phase 3: CLM + Contrastive
    # =====================================================================
    def train_phase3_epoch(self) -> float:
        """Train one epoch with CLM + contrastive loss."""
        self.model.train()
        loader = self._make_loader(self.train_dataset)
        total_loss, total_clm, total_ctr = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(loader, desc="  Phase 3 — CLM+Contrastive", unit="step", ncols=120, leave=True)
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)
            attention_mask = self._make_attention_mask(batch)
            token_mask = (batch != self.pad_token_id)  # (B, T) for mean pooling

            with autocast(device_type='cuda', enabled=self.use_fp16):
                # Pass 1
                _, clm_loss, hidden1, _ = self.model(
                    batch, labels=batch, attention_mask=attention_mask,
                )
                pooled1 = mean_pool(hidden1, token_mask)

                # Pass 2 (different dropout → different representation)
                _, _, hidden2, _ = self.model(
                    batch, attention_mask=attention_mask,
                )
                pooled2 = mean_pool(hidden2, token_mask)

                # Contrastive loss
                ctr_loss = self.contrastive_loss_fn(pooled1, pooled2)

                # Combined loss
                loss = clm_loss + self.contrastive_weight * ctr_loss

            self._optimizer_step(loss)
            self._step_if_ready(batch_idx)

            total_loss += loss.item()
            total_clm += clm_loss.item()
            total_ctr += ctr_loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "clm": f"{total_clm/num_batches:.4f}",
                "ctr": f"{total_ctr/num_batches:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        return total_loss / num_batches

    # =====================================================================
    # Validation
    # =====================================================================
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        loader = self._make_loader(self.val_dataset, shuffle=False)
        total_loss, num_batches = 0.0, 0

        pbar = tqdm(loader, desc="  Validation", unit="batch", ncols=100, leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            attention_mask = self._make_attention_mask(batch)

            with autocast(device_type='cuda', enabled=self.use_fp16):
                _, loss, _, _ = self.model(batch, labels=batch, attention_mask=attention_mask)

            total_loss += loss.item()
            num_batches += 1

            avg = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg:.4f}", "ppl": f"{math.exp(min(avg, 10)):.1f}"})

        return total_loss / num_batches

    # =====================================================================
    # Main loop
    # =====================================================================
    def train(self):
        print(f"\n{'='*80}")
        print("STARTING MULTI-PHASE TRAINING")
        print(f"{'='*80}\n")

        patience = self.config["training"]["early_stopping_patience"]
        checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        global_epoch = 0
        early_stopped = False

        phases = [
            ("Phase 1 — Denoising CLM", self.phase1_epochs, "phase1"),
            ("Phase 2 — Curriculum CLM", self.phase2_epochs, "phase2"),
            ("Phase 3 — CLM + Contrastive", self.phase3_epochs, "phase3"),
        ]

        for phase_name, phase_epochs, phase_key in phases:
            print(f"\n{'='*80}")
            print(f"  {phase_name}  ({phase_epochs} epochs)")
            print(f"{'='*80}\n")

            for epoch_in_phase in range(phase_epochs):
                global_epoch += 1
                epoch_start = time.time()

                # Train
                if phase_key == "phase1":
                    train_loss = self.train_phase1_epoch()
                elif phase_key == "phase2":
                    train_loss = self.train_phase2_epoch(epoch_in_phase)
                else:
                    train_loss = self.train_phase3_epoch()

                # Validate
                val_loss = self.validate()
                ppl = math.exp(min(val_loss, 10))
                elapsed = time.time() - epoch_start

                print(f"\n  Epoch {global_epoch}/{self.total_epochs} ({phase_name})")
                print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}  |  PPL: {ppl:.2f}  |  Time: {elapsed:.0f}s")

                # Dashboard
                if DASHBOARD_AVAILABLE and log_training_step:
                    try:
                        log_training_step(
                            epoch=global_epoch,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            perplexity=ppl,
                            learning_rate=self.scheduler.get_last_lr()[0],
                            epoch_time=elapsed,
                            gradient_norm=0.0,
                        )
                    except Exception:
                        pass

                # Early stopping
                if val_loss < self.best_val_loss:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    self.best_epoch = global_epoch
                    self.patience_counter = 0
                    self.model.save_pretrained(str(checkpoint_dir / "best_model.pt"))
                    print(f"  -> New best! (improved {improvement:.4f})  Patience: 0/{patience}")
                else:
                    self.patience_counter += 1
                    print(f"  -> No improvement.  Patience: {self.patience_counter}/{patience}")
                    if self.patience_counter >= patience:
                        print(f"\n  EARLY STOPPING at epoch {global_epoch}")
                        early_stopped = True
                        break

                # Save phase checkpoint
                save_interval = self.config["training"].get("save_interval_epochs", 2)
                if global_epoch % save_interval == 0:
                    self.model.save_pretrained(str(checkpoint_dir / f"checkpoint_epoch_{global_epoch}.pt"))

            if early_stopped:
                break

        # Final save
        self.model.save_pretrained(str(checkpoint_dir / "final_model.pt"))

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        if early_stopped:
            print(f"Stopped early at epoch {global_epoch}/{self.total_epochs}")
        print(f"Best model: epoch {self.best_epoch} — val_loss={self.best_val_loss:.4f}")
        print(f"Saved at: {checkpoint_dir / 'best_model.pt'}")
        print(f"{'='*80}\n")


def main():
    config_path = PROJECT_ROOT / "config" / "gpt2_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config["system"]["device"] == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["system"]["device"] = "cpu"

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = MultiPhaseTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
