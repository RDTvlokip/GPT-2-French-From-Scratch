# GPT-2 From Scratch — Modern Training Framework

A complete GPT-2 implementation from scratch in PyTorch with a **LLaMA-style modernized architecture** and a **multi-phase training pipeline**. Bring your own dataset, train your own model.

> **No HuggingFace `transformers` dependency.** Every component is implemented from first principles: tokenizer, model, attention, training loop, multi-phase curriculum. Use it for any language, any domain.

---

## What this is

A training framework you can use **with your own text data** (any language, any domain) to train a GPT-2 model from scratch with modern improvements:

- **Modern architecture**: RoPE, RMSNorm, SwiGLU, QK-Norm, Flash Attention — same building blocks as LLaMA/Mistral
- **3-phase training pipeline**: denoising CLM → curriculum learning → contrastive learning
- **Hardware-friendly**: runs on a single consumer GPU (8 GB VRAM minimum, tested on GTX 1080 Ti)
- **End-to-end pipeline**: from raw `.txt` files to a working generative model
- **Configurable**: every hyperparameter exposed in a single YAML file

The goal: **understand every line** of the code while building something that actually works.

---

## Features

### Architecture (all toggleable)
- **Pre-norm transformer** with weight tying
- **RoPE** (Rotary Position Embeddings)
- **RMSNorm** (~10% faster than LayerNorm)
- **SwiGLU MLP**
- **QK-Norm** for training stability
- **Flash Attention** via PyTorch 2.0+ SDPA
- **GQA** (Grouped Query Attention, optional)
- **KV-cache** for fast generation
- **Gradient checkpointing** (~60% VRAM savings)
- **Legacy GPT-2 mode** (disable all modern features, get vanilla GPT-2)

### Training
- Mixed precision (FP16) with automatic loss scaling
- Gradient accumulation
- Cosine LR schedule with warmup
- Gradient clipping
- Early stopping
- Multi-phase curriculum (see below)
- Label smoothing (optional)

### Tokenizer
- **ByteLevel BPE** (same family as GPT-2/3/4)
- Handles any Unicode (emojis, accents, code, etc.)
- 0% `<unk>` tokens by design
- Trainable on your own corpus

### Data pipeline
- Multiprocessing tokenization (memory-efficient numpy int32)
- Intelligent chunking with overlap
- Shuffled train/val split (no alphabetical bias)
- Markdown normalization utility

---

## Installation

```bash
git clone https://github.com/RDTvlokip/GPT-2-French-From-Scratch.git
cd GPT-2-French-From-Scratch
pip install -r requirements.txt
```

**Dependencies** (minimal):
- `torch >= 2.0`
- `tokenizers`
- `numpy`
- `pyyaml`
- `tqdm`

**Hardware**:
- Minimum: 8 GB VRAM (with gradient checkpointing enabled)
- Recommended: 11+ GB VRAM
- CPU: any modern CPU works, multiprocessing used for data prep

---

## Quick start (with your own data)

### 1. Drop your `.txt` files in `data/`

```
data/
├── document_001.txt
├── document_002.txt
└── ...
```

Any language. Any domain. One file per document. UTF-8 encoded.

### 2. Train a BPE tokenizer on YOUR corpus

```bash
python train_bpe.py
```

This builds `bpe_tokenizer_32k.json` adapted to your specific vocabulary. Edit `train_bpe.py` to change `VOCAB_SIZE` (default 32000).

### 3. Tokenize, chunk, and split your dataset

```bash
python scripts/prepare_data.py
```

Outputs `data/train.pt` and `data/val.pt` (90/10 shuffled split with seed=42). Uses multiprocessing — fast even on 100k+ files.

### 4. Configure your model

Edit `config/gpt2_config.yaml` — set the model size, training hyperparameters, and which modern features to enable.

### 5. Train

**Standard CLM** (single-phase):
```bash
python src/train.py
```

**Multi-phase pipeline** (recommended):
```bash
python src/train_multiphase.py
```

### 6. Generate

```bash
python src/generate.py --prompt "Your prompt here"
```

Or interactive mode:
```bash
python src/generate.py
```

---

## Configuration

Everything is in `config/gpt2_config.yaml`. Key sections:

```yaml
model:
  vocab_size: 32000
  n_positions: 1024              # Max context length
  n_embd: 256                    # Hidden dimension
  n_layer: 8                     # Number of transformer blocks
  n_head: 4                      # Number of attention heads

  # Modern features (toggle individually)
  use_rope: true
  use_rmsnorm: true
  use_swiglu: true
  use_qk_norm: true
  use_flash_attention: true
  use_gqa: false                 # n_kv_heads must divide n_head
  label_smoothing: 0.0           # 0.0 for small models, 0.1 for >50M params

training:
  batch_size: 16
  gradient_accumulation_steps: 4 # Effective batch = 16 × 4 = 64
  learning_rate: 5.0e-4
  weight_decay: 0.01
  lr_scheduler: "cosine"
  warmup_ratio: 0.07
  num_epochs: 20
  use_fp16: true
  gradient_checkpointing: false  # Enable to save VRAM
  early_stopping_patience: 3

multiphase:
  phase1_epochs: 3               # Denoising CLM
  phase2_epochs: 10              # Curriculum CLM
  phase3_epochs: 5               # CLM + Contrastive
  corruption_rate: 0.15          # 15% token corruption in phase 1
  num_curriculum_buckets: 5      # Difficulty buckets in phase 2
  contrastive_weight: 0.1        # Phase 3 loss weight
  contrastive_temperature: 0.05  # InfoNCE temperature

data:
  chunk_size: 450
  chunk_overlap: 50
  target_seq_length: 512
  train_split: 0.9
```

### Choosing model size

Follow the Chinchilla scaling law: aim for **~20 tokens per parameter**.

| Your dataset | Recommended model size |
|---|---|
| 10M tokens | ~500K params (very small) |
| 50M tokens | ~2.5M params |
| 200M tokens | ~10M params |
| 500M tokens | ~25M params |
| 1B tokens | ~50M params |
| 5B tokens | ~250M params |

Example presets (commented in `config/gpt2_config.yaml`):
- **Tiny** (n_embd=128, n_layer=2): 4.5M params
- **Small** (n_embd=256, n_layer=6): 13M params
- **Medium** (n_embd=512, n_layer=8): 44M params
- **GPT-2 Small** (n_embd=768, n_layer=12): 110M params

---

## Project structure

```
.
├── config/
│   └── gpt2_config.yaml          # All hyperparameters in one file
├── data/                         # YOUR text data goes here
│   ├── *.txt
│   ├── train.pt                  # Generated by prepare_data.py
│   └── val.pt
├── src/
│   ├── model/
│   │   └── gpt2.py               # Architecture (RoPE, RMSNorm, SwiGLU...)
│   ├── utils/
│   │   └── tokenizer.py          # BPE wrapper
│   ├── train.py                  # Standard CLM trainer
│   ├── train_multiphase.py       # Multi-phase trainer (recommended)
│   ├── curriculum.py             # Denoising + contrastive + scoring
│   └── generate.py               # Text generation with streaming
├── scripts/
│   ├── prepare_data.py           # Tokenize + chunk + split (parallel)
│   └── verify_setup.py           # Sanity checks
├── train_bpe.py                  # Train BPE tokenizer on your data
├── correct_markdown.py           # Optional markdown normalization
├── test_architecture.py          # Architecture test suite (10 tests)
└── README.md
```

---

## How the architecture works

### Attention block

```
x → Norm → [Q, K, V projection]
              ├─ QK-Norm (RMSNorm on Q and K)
              ├─ RoPE rotation (encodes position)
              └─ Flash Attention (causal + padding mask combined)
                 → output projection → + residual
```

### MLP block (SwiGLU)

```
x → Norm → [gate, up] = Linear(x).chunk(2)
           ↓
           SiLU(gate) * up      ← gated activation
           ↓
           down_proj → + residual
```

### Why these choices?

| Choice | Reason |
|---|---|
| **RoPE** | No learned positional embeddings, better length generalization, encodes relative position naturally |
| **RMSNorm** | ~10% faster than LayerNorm, fewer parameters, no centering bias |
| **SwiGLU** | Outperforms GELU at parameter parity (PaLM, LLaMA confirmed) |
| **QK-Norm** | Stabilizes training for small models, prevents attention logit explosion |
| **Flash Attention** | 2x speed + 4x VRAM savings via fused kernel |
| **Weight tying** | Saves vocab × n_embd params (embedding and lm_head share weights) |

All features are **toggleable**. Set them all to `false` in the config to get a vanilla GPT-2.

---

## How the multi-phase training works

### Phase 1 — Denoising CLM

The input is corrupted before being fed to the model:
- **50%** of selected tokens → replaced with a random token
- **25%** → replaced with `<unk>` (mask token)
- **25%** → kept original

The labels remain the **clean** original sequence. The model learns to predict the correct next token even from noisy context — building robust language understanding before tackling clean data.

### Phase 2 — Curriculum CLM

Training samples are scored by difficulty:
- Sequence length (longer = harder)
- Token diversity (more unique tokens = harder)
- Token rarity (rare tokens = harder)

Samples are split into N **cumulative** buckets:
- Bucket 1: easiest 20% only
- Bucket 2: easiest 40% (includes bucket 1)
- ...
- Bucket N: full dataset

Each epoch uses a progressively larger bucket. The model masters simple patterns before being exposed to complex ones.

### Phase 3 — CLM + Contrastive

For each batch, the model performs **two forward passes** with different dropout masks:

```
seq A → forward(dropout₁) → hidden_A1 → mean-pool → vec_A1
seq A → forward(dropout₂) → hidden_A2 → mean-pool → vec_A2
```

The contrastive loss (InfoNCE, T=0.05) pulls vec_A1 and vec_A2 together while pushing apart pairs from different sequences in the batch. This organizes the embedding space so semantically similar texts cluster together.

Total loss:
```
loss = CLM_loss + 0.1 × contrastive_loss
```

---

## Critical bugs found during development

These were caught and fixed. Reading them is more useful than any tutorial:

### 1. Flash Attention + padding mask broke causal masking
```python
# BUG: when attention_mask is provided, is_causal=False → model sees the future
is_causal=(past_key_value is None and attention_mask is None)
```
**Fix**: build a combined (causal + padding) mask manually when both are needed.

**Symptom**: PPL of 1.13 during training (impossible), generation produced repetitive garbage. The model was cheating by looking ahead.

### 2. Tokenizer auto-padded all batches to longest sequence
```python
# BUG: enable_padding() called globally in __init__
self.tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
```
**Fix**: `tokenizer.no_padding()` by default, pad explicitly where needed.

**Symptom**: dataset 12x larger than expected, 93% padding instead of actual content.

### 3. Loss computed on padding tokens
```python
# BUG: no ignore_index
loss = F.cross_entropy(shift_logits, shift_labels)
```
**Fix**: `ignore_index=pad_token_id`.

**Symptom**: artificially low validation loss because predicting `<pad>` after `<pad>` is trivial.

### 4. Label smoothing flattened distributions on small models
With `label_smoothing=0.1`, a 15M model's top token probability dropped to 0.3%, generating noise. Disabled by default for models <50M params.

### 5. `generate()` left model in `.eval()` mode
Calling `generate()` during training silently disabled dropout for subsequent training steps. Now restores the previous mode.

---

## Running the test suite

```bash
python test_architecture.py
```

10 tests covering:
- Legacy model (all features off)
- Each modern feature individually
- All features combined
- Flash vs manual attention equivalence
- Gradient checkpointing correctness
- KV-cache vs no-cache match
- VRAM and speed benchmarks

---

## Generation parameters

```bash
python src/generate.py \
  --prompt "Your prompt" \
  --max_length 512 \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.9 \
  --repetition_penalty 1.2
```

| Parameter | Effect |
|---|---|
| `temperature` | Higher = more random (0.7-0.9 typical) |
| `top_k` | Sample from top-k most likely tokens |
| `top_p` | Nucleus sampling (cumulative probability) |
| `repetition_penalty` | >1.0 discourages repeating tokens |
| `do_sample` | False = greedy decoding |

KV-cache is enabled by default (5-10x speedup).

---

## Example: what to expect

Trained on **~270M tokens** of structured French text, a 15M parameter model with this framework produces:

- ✅ Fluent grammar and syntax in the target language
- ✅ Markdown structure when the data contains it (titles, lists, tables, code blocks)
- ✅ Technical vocabulary appropriate to the training domain
- ✅ Zero spelling mistakes (thanks to ByteLevel BPE + clean data)
- ⚠️ Some factual hallucinations (expected at this scale)
- ⚠️ Topic drift after ~200 tokens (small model limitation)

For factual generation, scale up parameters (100M+) and/or use RAG.

---

## Hardware notes

Reference: GTX 1080 Ti (Pascal, 2017, no Tensor Cores):
- ~4-4.5 steps/sec with batch_size=16, FP16
- VRAM usage: ~5 GB for a 15M model
- Full multi-phase run: ~50-60 hours on 270M tokens

On modern hardware (RTX 4090, A100, H100), expect 5-10x speedup.

---

## Tips for getting good results

1. **Clean your data first**. A 100M token clean dataset beats a 1B token noisy one. Use `correct_markdown.py` if your data is structured markdown.

2. **Train your own tokenizer**. Don't reuse one trained on a different corpus — `train_bpe.py` adapts to your specific vocabulary.

3. **Respect Chinchilla scaling**. Don't make the model too big for your dataset. Use the table above.

4. **Enable all modern features by default**. They cost nothing and help on every dataset size.

5. **Start with `train.py` to validate the pipeline**. Once it works, switch to `train_multiphase.py` for better final quality.

6. **Monitor PPL, not just loss**. A PPL of 10-30 means the model understands. A PPL near 1 means it's memorizing or has a bug.

7. **Check generation quality, not just metrics**. Loss can be misleading (see bug #3 above).

---

## License

**AGPL-3.0** — see [LICENSE](LICENSE) for the full text.

This is a strong copyleft license. Key implications:
- You can use, modify, and distribute this code freely
- Any derivative work **must** be released under AGPL-3.0
- If you run a modified version as a network service (web app, API, SaaS), you **must** provide the source code to your users
- Commercial use is allowed, but the copyleft obligation applies

---

## Acknowledgements

- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) for the spiritual ancestor
- The LLaMA papers (Touvron et al.) for the architectural blueprint
- The PyTorch team for Flash Attention via SDPA

---

## Contact

Issues, questions, contributions: open an issue on the repository.

Built solo, on a GTX 1080 Ti, because we don't all have $30,000 GPUs — but we can still understand what's inside the box.