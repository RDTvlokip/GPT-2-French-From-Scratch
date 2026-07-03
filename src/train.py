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

# Force UTF-8 on the console so progress symbols (✓ ✗ →) and French accents
# print correctly on Windows (cp1252) instead of crashing or showing "�".
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

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



class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al. 2023, "Symbolic Discovery of Optimization
    Algorithms"). Update = sign of an interpolation of the momentum, so update
    magnitude is constant -> needs a LR ~3-10x SMALLER than AdamW and often a
    larger weight_decay. Simpler and lighter than Adam (1 state tensor vs 2).

    update_t = sign( (1-β1)·g_t + β1·m_{t-1} )
    m_t      = (1-β2)·g_t + β2·m_{t-1}
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                # decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = m.mul(b1).add_(g, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)
        return loss


def compute_gradient_stats(model):
    """Compute the global gradient L2 norm (logged per epoch in metrics.csv)."""
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

    def __init__(self, data_path: str, fraction: float = 1.0, seed: int = 42):
        """
        Initialize dataset.

        Args:
            data_path: Path to .pt file containing tokenized data
            fraction: Fraction of blocks to keep (1.0 = all). A random subset is
                drawn with a fixed seed for reproducible quick test runs.
            seed: RNG seed used when fraction < 1.0.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.data = torch.load(data_path, weights_only=True)

        if fraction < 1.0:
            n_total = len(self.data)
            n_keep = max(1, int(n_total * fraction))
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(n_total, generator=g)[:n_keep]
            self.data = self.data[idx]
            print(f"Loaded dataset from {data_path} (subset {fraction:.0%}: "
                  f"{n_keep:,}/{n_total:,} blocks)")
        else:
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


class _Tee:
    """Duplicate a stream to a log file (so console output is also saved).

    Wraps sys.stdout/sys.stderr: everything printed still shows on the console
    AND is appended to the run's .log file. No need to copy-paste logs anymore.
    """

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        # Never let a console encoding issue (e.g. cp1252 can't encode '✓' or
        # '→') crash training. Fall back to an ASCII-safe version on the console;
        # the log file is UTF-8 so it always keeps the original text.
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            enc = getattr(self.stream, "encoding", "ascii") or "ascii"
            self.stream.write(data.encode(enc, errors="replace").decode(enc))
        try:
            self.log_file.write(data)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        self.stream.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        # Delegate anything else (encoding, isatty, ...) to the real stream.
        return getattr(self.stream, name)


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
        # Device type ("cuda"/"cpu") for autocast/GradScaler; handles "cuda:0" form
        self.device_type = torch.device(self.device).type

        # Create directories
        os.makedirs(config["logging"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["logging"]["log_dir"], exist_ok=True)

        # Automatic logging: each run gets its OWN dated subfolder under log_dir,
        # holding the full console log and the per-epoch metrics CSV. So runs
        # never overwrite each other and everything for one run lives together.
        # The folder name is auto-built from the actual run params (vocab,
        # param count, epochs, % of data) so it always matches reality.
        log_dir = Path(config["logging"]["log_dir"])
        run_name = self._build_run_name(config)
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._run_log_dir = log_dir / f"{stamp}_{run_name}"
        os.makedirs(self._run_log_dir, exist_ok=True)

        self._log_fh = open(self._run_log_dir / "console.log", "w", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, self._log_fh)
        sys.stderr = _Tee(sys.stderr, self._log_fh)

        self._metrics_path = self._run_log_dir / "metrics.csv"
        with open(self._metrics_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,perplexity,learning_rate,epoch_time,grad_norm\n")
        print(f"[Logging] Run folder -> {self._run_log_dir}")
        print(f"[Logging]   console.log + metrics.csv")

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

        # Initialize datasets. data_fraction < 1.0 trains on a random subset
        # (for fast validation runs); validation always uses the full set.
        print("\nLoading datasets...")
        data_fraction = float(config["data"].get("data_fraction", 1.0))
        seed = config["training"].get("seed", 42)
        self.train_dataset = TokenDataset(
            config["data"]["train_data"], fraction=data_fraction, seed=seed
        )
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

        # Initialize optimizer. Selectable via config (optimizer: adamw|lion)
        # to experiment with the "how it learns" axis.
        opt_name = config["training"].get("optimizer", "adamw").lower()
        lr = config["training"]["learning_rate"]
        wd = config["training"]["weight_decay"]
        if opt_name == "lion":
            # Lion needs a much smaller LR than AdamW (sign-based updates).
            beta1 = config["training"].get("lion_beta1", 0.9)
            beta2 = config["training"].get("lion_beta2", 0.99)
            self.optimizer = Lion(self.model.parameters(), lr=lr,
                                  betas=(beta1, beta2), weight_decay=wd)
            print(f"Optimizer: Lion (lr={lr}, betas=({beta1},{beta2}), wd={wd}) "
                  f"-- ATTENTION: Lion veut un LR ~3-10x plus petit qu'AdamW")
        else:
            beta1 = config["training"].get("adam_beta1", 0.9)
            beta2 = config["training"].get("adam_beta2", 0.999)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd,
                betas=(beta1, beta2),
            )
            print(f"Optimizer: AdamW (lr={lr}, betas=({beta1},{beta2}), wd={wd})")

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
        self.scaler = GradScaler(device=str(self.device)) if self.use_fp16 else None

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
            with autocast(device_type=self.device_type, enabled=self.use_fp16):
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

            # Add GPU memory if CUDA available. Report the PEAK reserved memory
            # (what PyTorch actually holds from the driver, close to nvidia-smi),
            # not memory_allocated() which only counts live tensors at this
            # instant and badly under-reports real usage.
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_reserved() / 1e9
                postfix_dict["gpu"] = f"{peak_gb:.1f}GB"

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

            with autocast(device_type=self.device_type, enabled=self.use_fp16):
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

    @staticmethod
    def _build_run_name(config: dict) -> str:
        """Build a self-describing run name from the actual run parameters.

        Format: gpt_<vocab>_<params>_<epochs>epochs_<pct>data_v1
        e.g. gpt_32k_15m_6epochs_60data_v1
        """
        m = config["model"]
        vocab = m["vocab_size"]
        vocab_str = f"{vocab // 1000}k" if vocab % 1000 == 0 else str(vocab)

        # Estimate parameter count from the config (tied embeddings: the
        # vocab/embedding matrix is counted once). This matches GPT2.get_num_params
        # closely enough for a label.
        n_embd, n_layer = m["n_embd"], m["n_layer"]
        n_inner = m.get("n_inner", 4 * n_embd)
        # per-layer: attention (~4 * n_embd^2) + MLP (~2 * n_embd * n_inner)
        per_layer = 4 * n_embd * n_embd + 2 * n_embd * n_inner
        embed = vocab * n_embd  # tied with lm_head, counted once
        total = embed + n_layer * per_layer
        if total >= 1_000_000:
            params_str = f"{round(total / 1_000_000)}m"
        else:
            params_str = f"{round(total / 1000)}k"

        epochs = config["training"]["num_epochs"]
        frac = config["data"].get("data_fraction", 1.0)
        pct = round(frac * 100)

        return f"gpt_{vocab_str}_{params_str}_{epochs}epochs_{pct}data_v1"

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
            # Also keep a copy alongside this run's logs/metrics, so each run
            # folder is self-contained (model + console.log + metrics.csv).
            self.model.save_pretrained(str(self._run_log_dir / "best_model.pt"))
            # No print here - we print in the train() method with more context

        # Save intermediate checkpoint
        if save_interval_epochs and epoch % save_interval_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            self.model.save_pretrained(str(checkpoint_path))

    def save_resume_state(self, epoch: int):
        """Save a FULL training state so a run can resume exactly after a crash
        (power outage, Ctrl+C...). Unlike best_model.pt (weights only), this
        keeps the optimizer, scheduler, epoch, best-val tracking and RNG state.
        Overwrites resume.pt each epoch (single rolling file).
        """
        checkpoint_dir = Path(self.config["logging"]["checkpoint_dir"])
        resume_path = checkpoint_dir / "resume.pt"
        state = {
            "epoch": epoch,                       # last COMPLETED epoch (1-based)
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": getattr(self, "global_step", 0),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        # Write to a temp file then rename: a crash mid-write can't corrupt the
        # existing resume.pt.
        tmp = checkpoint_dir / "resume.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, resume_path)

    def load_resume_state(self):
        """Load resume.pt if present. Returns the epoch to start from (0 if none)."""
        resume_path = Path(self.config["logging"]["checkpoint_dir"]) / "resume.pt"
        if not resume_path.exists():
            return 0
        print(f"\n[Resume] Found {resume_path}, restoring full training state...")
        ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_epoch = ckpt["best_epoch"]
        self.patience_counter = ckpt["patience_counter"]
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
            if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        except Exception:
            pass
        start = ckpt["epoch"]  # resume AFTER this completed epoch
        print(f"[Resume] Resuming from epoch {start + 1} "
              f"(best val_loss so far: {self.best_val_loss:.4f} at epoch {self.best_epoch})")
        return start

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")

        num_epochs = self.config["training"]["num_epochs"]
        patience = self.config["training"]["early_stopping_patience"]
        early_stopped = False

        # Resume from a previous crash if resume.pt exists (full state restored).
        start_epoch = self.load_resume_state()

        epochs_pbar = tqdm(range(start_epoch, num_epochs), desc="Training",
                           unit="epoch", colour="green")

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

            # Append structured metrics row (for plotting / comparing runs).
            current_lr = self.scheduler.get_last_lr()[0]
            grad_norm = compute_gradient_stats(self.model)
            try:
                with open(self._metrics_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},"
                            f"{perplexity:.4f},{current_lr:.6e},{epoch_time:.1f},{grad_norm:.4f}\n")
            except Exception as e:
                print(f"[Logging] Metrics write skipped (non-critical): {e}")

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

            # Save full resume state after each completed epoch (crash-safe).
            self.save_resume_state(epoch=epoch + 1)

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
        print(f"Logs & metrics:      {self._run_log_dir}")
        print("=" * 80 + "\n")

        # Restore original streams and close the log file handle.
        self._close_logging()

    def _close_logging(self):
        """Restore stdout/stderr and close the run log file."""
        if isinstance(sys.stdout, _Tee):
            sys.stdout = sys.stdout.stream
        if isinstance(sys.stderr, _Tee):
            sys.stderr = sys.stderr.stream
        try:
            self._log_fh.close()
        except Exception:
            pass


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Train GPT-2 from scratch")
    parser.add_argument(
        "--data-fraction", type=float, default=None,
        help="Fraction of training blocks to use (e.g. 0.2 for a fast test "
             "run). Overrides data.data_fraction in the config.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training.num_epochs (e.g. 1 for a quick run).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a config YAML (default: config/gpt2_config.yaml).",
    )
    args = parser.parse_args()

    # Load config (works from root or src/)
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / args.config
    else:
        config_path = PROJECT_ROOT / "config" / "gpt2_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides (take precedence over the config file)
    if args.data_fraction is not None:
        config["data"]["data_fraction"] = args.data_fraction
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs

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