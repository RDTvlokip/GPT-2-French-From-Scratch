"""Balayage des paramètres de sampling pour trouver le meilleur réglage de
génération d'un modèle donné, mesuré objectivement (rep3 / distinct2 /
coherence). Zéro re-train.

On teste une grille de (temperature, top_k, top_p, repetition_penalty) et on
compare chaque réglage à la config par défaut de la config YAML.

Usage:
    python scripts/tune_sampling.py --model models/best_model.pt
"""

import sys
import argparse
from pathlib import Path
from itertools import product

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

from model.gpt2 import GPT2
from utils.tokenizer import GPT2Tokenizer
from scripts.evaluate import (
    repetition_3gram, distinct_2, coherence_len, build_prompts,
)


@torch.no_grad()
def sample_generate(model, tokenizer, prompt_ids, device, max_new,
                    temperature, top_k, top_p, rep_pen, ban_bos=True):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_pos = model.config.n_positions
    gen = []
    for _ in range(max_new):
        cond = ids if ids.size(1) <= n_pos else ids[:, -n_pos:]
        logits = model(cond)[0][:, -1, :]
        if ban_bos:
            logits[:, tokenizer.bos_token_id] = float("-inf")
        # repetition penalty (HF semantics)
        if rep_pen != 1.0:
            seen = ids[0].tolist()
            for t in set(seen):
                v = logits[0, t]
                logits[0, t] = v / rep_pen if v > 0 else v * rep_pen
        logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[logits < kth] = float("-inf")
        if top_p:
            s_logits, s_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            rm = cum > top_p
            rm[..., 1:] = rm[..., :-1].clone()
            rm[..., 0] = 0
            # remettre le masque dans l'ordre original des tokens
            rm_orig = rm.scatter(1, s_idx, rm)
            logits[rm_orig] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tid = nxt.item()
        if tid == tokenizer.eos_token_id:
            break
        gen.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return gen


def eval_setting(model, tokenizer, prompts, device, drift, max_new,
                 temp, top_k, top_p, rep_pen, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reps, dists, cohs = [], [], []
    for pids in prompts:
        g = sample_generate(model, tokenizer, pids, device, max_new,
                            temp, top_k, top_p, rep_pen)
        if not g:
            continue
        reps.append(repetition_3gram(g))
        dists.append(distinct_2(g))
        cohs.append(coherence_len(g, drift))
    nz = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return round(nz(reps), 4), round(nz(dists), 4), round(nz(cohs), 1)


def main():
    ap = argparse.ArgumentParser(description="Tune sampling parameters")
    ap.add_argument("--model", default="models/best_model.pt")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "bpe_tokenizer_32k.json"))
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data" / "val.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=90)
    args = ap.parse_args()

    mp = args.model if Path(args.model).is_absolute() else str(PROJECT_ROOT / args.model)
    tokenizer = GPT2Tokenizer(args.tokenizer)
    model = GPT2.from_pretrained(mp, device=args.device); model.eval()

    drift = set()
    for s in ["#", " #", "##", " ##"]:
        for tid in tokenizer.encode(s, add_special_tokens=False):
            drift.add(tid)

    auto, fixed = build_prompts(tokenizer, args.val, args.n_prompts)
    prompts = auto + fixed
    print(f"Tuning on {len(prompts)} prompts\n")

    # Grille : on fait varier temperature, top_k, repetition_penalty.
    # top_p fixé à 0.9 (nucleus standard). Réglage actuel = repère.
    temps = [0.6, 0.7, 0.8]
    topks = [20, 40]
    reppens = [1.1, 1.3]
    top_p = 0.9

    print(f"{'temp  topk  rep':<18}{'rep3↓':>9}{'dist2↑':>9}{'coh↑':>8}")
    print("-" * 44)

    results = []
    for temp, top_k, rep in product(temps, topks, reppens):
        r, d, c = eval_setting(model, tokenizer, prompts, args.device, drift,
                               args.max_new, temp, top_k, top_p, rep)
        results.append(((temp, top_k, rep), r, d, c))
        print(f"{f't{temp} k{top_k} r{rep}':<18}{r:>9}{d:>9}{c:>8}")

    # Recommandation : meilleur compromis = faible rep3, bonne coherence.
    # Score simple : coherence - 100*rep3 (pénalise la répétition).
    best = max(results, key=lambda x: x[3] - 100 * x[1])
    print("\n" + "=" * 44)
    print(f"Meilleur compromis (coh - 100*rep3) : "
          f"temp={best[0][0]} top_k={best[0][1]} rep_pen={best[0][2]}")
    print(f"  rep3={best[1]} dist2={best[2]} coh={best[3]}")
    print("rep3: lower=better | dist2/coh: higher=better")


if __name__ == "__main__":
    main()
