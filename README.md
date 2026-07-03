# GPT-2 From Scratch — Modern Training Framework

A complete GPT-2 implementation from scratch in PyTorch with a **LLaMA-style modernized architecture** and a **multi-phase training pipeline**. Bring your own dataset, train your own model.

> **No HuggingFace `transformers` dependency.** Every component is implemented from first principles: tokenizer, model, attention, training loop, multi-phase curriculum. Use it for any language, any domain.

---

## 🤗 Pretrained models (French, 15M, from scratch)

Three recurrent-depth variants trained with this framework, on the Hub (Apache-2.0):

| Model | What it is | Strength |
|---|---|---|
| 🔁 [**Cadence**](https://huggingface.co/RDTvlokip/Cadence-15M-fr) | looped R=4 (the reference) | best perplexity (28.9) |
| 🎯 [**Focal**](https://huggingface.co/RDTvlokip/Focal-15M-fr) | + learned absolute-entropy halting | best in-domain coherence |
| 🧭 [**Nomade**](https://huggingface.co/RDTvlokip/Nomade-15M-fr) | + learned percentile halting (size-invariant) | best out-of-domain + fewest hallucinated names |

> ⚠️ **Preliminary — 1 seed, variance not controlled.** These are research artifacts, not benchmarked models.

**Write-ups**: [1 · I trained my own French LLM from scratch](https://huggingface.co/blog/RDTvlokip/i-trained-my-own-french-llm-from-scratch) · [2 · Architecture is a threshold, not a lever](https://huggingface.co/blog/RDTvlokip/what-i-learned-optimizing-a-15m-french) · [3 · Teaching a 15M LLM to think deeper](https://huggingface.co/blog/RDTvlokip/teaching-a-15m-french-llm-to-think-deeper)

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
- **🔁 Recurrent-depth / looped transformer** — apply the blocks `R` times
  (effective depth `n_layer × R`, **zero added parameters**) with a per-iteration
  depth embedding + zero-init residuals. Universal-Transformer / Huginn style.
- **🎯 Adaptive per-token halting** — tokens exit the loop early on an entropy
  criterion (absolute threshold **or** size-invariant percentile), learned at
  training time. 0 added params. See the *Recurrent-depth & adaptive halting* section below.
- **Legacy GPT-2 mode** (disable all modern features, get vanilla GPT-2)

### Training
- Mixed precision (FP16) with automatic loss scaling
- Gradient accumulation
- Cosine LR schedule with warmup
- Gradient clipping
- Early stopping
- Multi-phase curriculum (see below)
- Label smoothing (optional)
- **Automatic per-run logging** — each run gets a dated folder under `logs/`
  with `best_model.pt` + `console.log` + `metrics.csv`
- **`--data-fraction` / `--epochs`** flags for fast validation runs
- **Crash-safe resume** — full state (optimizer + scheduler + epoch + RNG) is
  written atomically to `<checkpoint_dir>/resume.pt`; a relaunched run continues
  where it stopped. *(Delete `resume.pt` to force a fresh run.)*

### Tokenizer
- **ByteLevel BPE** (same family as GPT-2/3/4)
- Handles any Unicode (emojis, accents, code, etc.)
- 0% `<unk>` tokens by design
- Trainable on your own corpus

### Data pipeline
- Multiprocessing tokenization (memory-efficient numpy int32)
- Document packing into full blocks (no padding, no mid-sentence cuts)
- Shuffled train/val split (no alphabetical bias)
- **Corpus cleaner** (`clean_corpus.py`) — strips emojis, normalizes special
  whitespace and quotes, **while protecting code blocks**
- Markdown normalization utility

### Evaluation & decoding
- **Measurement harness** (`evaluate.py`) — quantifies generation quality:
  perplexity, 3-gram repetition, distinct-2, `coherence_len` (tokens before
  topic drift), plus **4 anti-gaming metrics** (`self_ppl`, `proper_noun_ratio`
  for hallucinated names, `prompt_overlap`, `burstiness`) that catch decoding
  tricks `coherence_len` alone would miss. Reproducible (fixed seed), one dated
  folder per eval. Halting flags: `--adaptive-halting --halting-mode ...`.
- **Sampling tuner** (`tune_sampling.py`) — sweeps temperature / top_k /
  repetition_penalty against the harness to find the best decoding config
- **Contrastive decoding** (`contrastive_decode.py`) and **DoLa**
  (`dola_decode.py`) — Li et al. 2022 / Chuang et al. 2023, for experiments
- **On-topic decoding** — strips the trailing `<eos>` from the prompt and bans
  `<bos>` mid-generation so the model continues instead of starting a new doc

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

### 2. (Optional) Clean your corpus

```bash
python scripts/clean_corpus.py
```

Strips decorative emojis, normalizes special whitespace (narrow no-break
spaces, etc.) and straight quotes `"` into French `« »`, **while protecting
code blocks** (anything in ``` fences or `inline` backticks is left untouched).
Edits files in place — back up `data/` first. Idempotent and interruptible.

### 3. Train a BPE tokenizer on YOUR corpus

```bash
python scripts/train_bpe.py
```

This builds `bpe_tokenizer_32k.json` adapted to your specific vocabulary. Edit `scripts/train_bpe.py` to change `VOCAB_SIZE` (default 32000).

### 4. Tokenize and pack your dataset

```bash
python scripts/prepare_data.py
```

Uses **document packing** (GPT-3 / LLaMA style): each document is wrapped in
`<bos> … <eos>`, all documents are concatenated into one continuous token
stream, then sliced into full blocks of `block_size` (default 768) — **no
padding, no mid-sentence cuts**. Outputs `data/train.pt` and `data/val.pt`
(90/10 shuffled split). Multiprocessing — fast even on 100k+ files.

### 5. Configure your model

Edit `config/gpt2_config.yaml` — set the model size, training hyperparameters, and which modern features to enable.

### 6. Train

**Standard CLM** (single-phase):
```bash
python src/train.py
```

Useful flags for quick validation runs:
```bash
python src/train.py --data-fraction 0.2 --epochs 3   # train on 20% of blocks
```

**Multi-phase pipeline**:
```bash
python src/train_multiphase.py
```

Each run writes a self-contained, auto-named folder under `logs/`
(e.g. `logs/2026-06-29_…_gpt_32k_14m_6epochs_60data_v1/`) holding
`best_model.pt`, the full `console.log`, and `metrics.csv` (one row per epoch).

### 7. Generate

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
  block_size: 768                # Training sequence length (<= n_positions)
  train_split: 0.9               # 90% train, 10% validation
  data_fraction: 1.0             # Fraction of train blocks (1.0 = all)
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
│   ├── gpt2_config.yaml                # Main config (all hyperparameters)
│   ├── looped_config.yaml             # Cadence — looped R=4
│   ├── adaptive_looped_config.yaml    # Focal — + absolute-entropy halting
│   ├── percentile_looped_config.yaml  # Nomade — + percentile halting
│   ├── baseline15m_config.yaml        # 15M vanilla baseline (comparison)
│   ├── scaleup_config.yaml            # 26.5M scale-up
│   ├── amateur_config.yaml            # 4.5M (for contrastive decoding)
│   ├── layerscale_config.yaml         # experiment: LayerScale
│   ├── multitoken_config.yaml         # experiment: multi-token prediction
│   ├── optim_beta2_config.yaml        # experiment: AdamW β2=0.95
│   └── optim_lion_config.yaml         # experiment: Lion optimizer
├── data/                         # YOUR text data goes here
│   ├── *.txt
│   ├── train.pt                  # Generated by prepare_data.py (packed blocks)
│   └── val.pt
├── src/
│   ├── model/
│   │   └── gpt2.py               # Architecture (RoPE, RMSNorm, SwiGLU...)
│   ├── utils/
│   │   └── tokenizer.py          # BPE wrapper
│   ├── train.py                  # Standard CLM trainer (auto logging + metrics)
│   ├── train_multiphase.py       # Multi-phase trainer
│   ├── curriculum.py             # Denoising + contrastive + scoring
│   └── generate.py               # Text generation with streaming
├── scripts/
│   ├── prepare_data.py           # Tokenize + document packing (parallel)
│   ├── clean_corpus.py           # Strip emojis, fix quotes/spaces (parallel)
│   ├── train_bpe.py              # Train BPE tokenizer on your data
│   ├── count_tokens.py           # Count tokens across the dataset
│   ├── correct_markdown.py       # Optional markdown normalization
│   ├── verify_setup.py           # Sanity checks
│   ├── evaluate.py               # Measurement harness (7 metrics, anti-gaming)
│   ├── tune_sampling.py          # Sweep temperature / top_k / repetition_penalty
│   ├── best_of_n.py              # Best-of-N re-scored by self_ppl
│   ├── contrastive_decode.py     # Contrastive decoding (Li et al. 2022)
│   ├── dola_decode.py            # DoLa layer contrast (Chuang et al. 2023)
│   └── entropy_sampling.py       # Entropy-adaptive sampling
├── tests/
│   ├── test_architecture.py      # Architecture test suite (10 tests)
│   ├── test_generation.py        # Generation smoke test
│   └── test_tokenizer.py         # Tokenizer test
├── logs/                         # One auto-named folder per run
│   └── <date>_gpt_…/             # best_model.pt + console.log + metrics.csv
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

## 🔁 Recurrent-depth & adaptive halting (research)

Instead of stacking `L` distinct layers, **loop the same `L` blocks `R` times** —
effective depth `L × R` with **zero added parameters** (beyond a tiny per-iteration
depth embedding). Then let each token **decide how deep to think**: it exits the
loop early once its output entropy is low enough. Both are toggled in the config:

```yaml
model:
  recurrence: 4                    # loop the blocks R times (1 = plain model)
  zero_init_residual: true         # blocks start as identity — stable unrolling
  adaptive_halting: true           # per-token early exit on entropy
  halting_in_training: true        # learn to be good at variable depth (Option B)
  halting_mode: percentile         # "absolute" (fixed nats) | "percentile" (q, size-invariant)
  halting_percentile: 0.3          # q: freeze the 30% least-uncertain active tokens / iter
```

Ready-made configs: `config/looped_config.yaml` (Cadence), `config/adaptive_looped_config.yaml`
(Focal), `config/percentile_looped_config.yaml` (Nomade).

**Results at 15M (robust eval, 50 prompts × 200 tokens, 1 seed — preliminary).**
Auto = in-domain (held-out corpus) prompts; Fixed = out-of-domain hand-written prompts:

| Model | Val PPL ↓ | Coherence auto ↑ | Coherence fixed ↑ | Invented names auto ↓ |
|---|---|---|---|---|
| Baseline (vanilla) | 31.2 | 35.2 | 40.7 | 0.137 |
| Cadence (looped R=4) | **28.9** | 39.3 | 32.5 | 0.121 |
| Focal (absolute halt) | 29.1 | **44.1** | 29.1 | 0.103 |
| Nomade (percentile halt) | 31.0 | 36.3 | **41.8** | **0.094** |

**No universal winner** — recurrence helps in-domain and hurts out-of-domain;
percentile halting trades the other way. Factuality is *not* improved (15M ceiling):
these are **quality** levers (coherence, fewer invented names), not capacity.

⚠️ Two traps worth knowing (both cost hours): **(a)** don't combine `zero_init_residual`
with LayerScale — the sub-block outputs 0, so LayerScale's γ never gets a gradient;
**(b)** halting only helps if it's **learned** — bolting it onto a fixed-R model at
inference under-computes.

**Lineage** (this is not novel): ACT ([Graves 2016](https://arxiv.org/abs/1603.08983)),
Universal Transformer ([Dehghani 2018](https://arxiv.org/abs/1807.03819)),
Recurrent-Depth Transformer / Huginn ([Geiping 2025](https://arxiv.org/abs/2502.05171)),
CALM ([Schuster 2022](https://arxiv.org/abs/2207.07061)),
LoopViT ([Shu 2026](https://arxiv.org/abs/2602.02156)).

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

### 6. Trailing `<eos>` on the prompt made the model change subject
```python
# BUG: the tokenizer's post-processor wraps every encode in <bos> $A <eos>,
# so encoding a prompt for generation appended <eos> at the end.
input_ids = tokenizer.encode(prompt, add_special_tokens=True)  # ... <eos>
```
With document packing (`...<eos><bos># New title...`), the model had learned
that an `<eos>` marks the end of a document. A prompt ending in `<eos>` told it
"this document is finished", so instead of continuing the prompt it started a
brand new document — emitting `<bos>` then a `# New title` and dropping the
subject entirely.

**Fix**: at inference, keep the leading `<bos>` but strip the trailing `<eos>`
(the prompt isn't finished — we want to *continue* it). As an extra guard, ban
the `<bos>` logit during decoding (`no_new_doc`, on by default; disable with
`--allow-new-doc`) so the model can't start a new document mid-generation.

**Symptom**: prompt "…in the town of Saint-Céré, in the Lot department" →
"… # The commune of Évreux, nestled in the French Alps …". The model wrote
fluent, coherent prose — about a completely different place. Train-time special
tokens (`<bos> text <eos>`) are NOT what you want at inference (`<bos> text…`).

---

## Running the test suite

```bash
python tests/test_architecture.py
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
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.9 \
  --repetition_penalty 1.3
```

| Parameter | Effect |
|---|---|
| `temperature` | Higher = more random (0.7-0.9 typical) |
| `top_k` | Sample from top-k most likely tokens |
| `top_p` | Nucleus sampling (cumulative probability) |
| `repetition_penalty` | >1.0 discourages repeating tokens — **1.3 doubled coherence here** (see `tune_sampling.py`) |
| `do_sample` | False = greedy decoding (small models repeat badly in greedy — keep sampling on) |
| `--allow-new-doc` | Let the model start a new document mid-generation (off by default; see bug #6) |

KV-cache is enabled by default (5-10x speedup). By default the generator strips
the trailing `<eos>` from the prompt and bans `<bos>` during decoding, so the
model continues your prompt instead of jumping to a new document (see bug #6).

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

1. **Clean your data first**. A 100M token clean dataset beats a 1B token noisy one. Run `scripts/clean_corpus.py` to strip emojis and normalize quotes/spaces; `scripts/correct_markdown.py` for structured markdown.

2. **Train your own tokenizer**. Don't reuse one trained on a different corpus — `scripts/train_bpe.py` adapts to your specific vocabulary.

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
