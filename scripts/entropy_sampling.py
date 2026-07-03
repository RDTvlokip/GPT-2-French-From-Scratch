"""Entropy-adaptive sampling : température dynamique par token, fonction de la
confiance du modèle (entropie des logits).

Idée : au lieu d'une température fixe, on l'ajuste à chaque pas.
   temp_eff = temp_base * (1 + k * H_norm)
où H_norm ∈ [0,1] est l'entropie normalisée des logits (après top_k) :
   H_norm = H(probs) / log(top_k)   (0 = très confiant, 1 = uniforme)

- k > 0  : "classique" — modèle hésitant (H haut) -> température PLUS haute
           (on explore dans l'incertitude au lieu de forcer un choix douteux).
- k < 0  : "inverse" — modèle hésitant -> température PLUS basse
           (on se resserre quand il ne sait pas).
- k = 0  : baseline (température fixe = réglage optimal trouvé).

On balaie k des DEUX côtés (à petite échelle l'intuition standard peut être
fausse — on mesure). Comparaison via le harnais evaluate.py, même seed.

Usage:
    python scripts/entropy_sampling.py --model models/best_model.pt
"""

import sys
import math
import argparse
from pathlib import Path

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
def adaptive_generate(model, tokenizer, prompt_ids, device, max_new,
                      temp_base=0.8, top_k=40, rep_pen=1.3, k=0.0,
                      ban_bos=True):
    """Génère avec température adaptée à l'entropie. k=0 -> baseline temp fixe."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_pos = model.config.n_positions
    log_topk = math.log(top_k) if top_k and top_k > 1 else 1.0
    gen = []
    for _ in range(max_new):
        cond = ids if ids.size(1) <= n_pos else ids[:, -n_pos:]
        logits = model(cond)[0][:, -1, :]
        if ban_bos:
            logits[:, tokenizer.bos_token_id] = float("-inf")
        if rep_pen != 1.0:
            for t in set(ids[0].tolist()):
                v = logits[0, t]
                logits[0, t] = v / rep_pen if v > 0 else v * rep_pen

        # top_k d'abord (l'entropie est mesurée sur le pool effectif)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        # entropie normalisée du pool restant (à température 1)
        p = F.softmax(logits, dim=-1)
        H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)  # (1,)
        H_norm = (H / log_topk).clamp(0.0, 1.0).item()

        temp_eff = max(0.05, temp_base * (1.0 + k * H_norm))
        probs = F.softmax(logits / temp_eff, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tid = nxt.item()
        if tid == tokenizer.eos_token_id:
            break
        gen.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return gen


def eval_k(model, tokenizer, prompts, device, drift, max_new, k, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reps, dists, cohs = [], [], []
    for pids in prompts:
        g = adaptive_generate(model, tokenizer, pids, device, max_new, k=k)
        if not g:
            continue
        reps.append(repetition_3gram(g))
        dists.append(distinct_2(g))
        cohs.append(coherence_len(g, drift))
    nz = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return round(nz(reps), 4), round(nz(dists), 4), round(nz(cohs), 1)


def main():
    ap = argparse.ArgumentParser(description="Entropy-adaptive sampling sweep")
    ap.add_argument("--model", default="models/best_model.pt")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "bpe_tokenizer_32k.json"))
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data" / "val.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=90)
    ap.add_argument("--ks", default="-0.5,-0.25,0.25,0.5,1.0",
                    help="facteurs k à tester (k=0 = baseline, ajouté auto)")
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
    print(f"Entropy-adaptive sampling on {len(prompts)} prompts")
    print("temp_eff = 0.8 * (1 + k * H_norm) | k>0: hésitant->chaud, k<0: hésitant->froid\n")

    print(f"{'k':<10}{'rep3↓':>9}{'dist2↑':>9}{'coh↑':>8}")
    print("-" * 36)
    r, d, c = eval_k(model, tokenizer, prompts, args.device, drift, args.max_new, 0.0)
    print(f"{'0 (base)':<10}{r:>9}{d:>9}{c:>8}")
    base = (r, d, c)

    ks = sorted(float(x) for x in args.ks.split(","))
    for k in ks:
        r, d, c = eval_k(model, tokenizer, prompts, args.device, drift, args.max_new, k)
        mark = ""
        if c > base[2]:
            mark += " coh↑"
        if r < base[0]:
            mark += " rep↓"
        print(f"{k:<10}{r:>9}{d:>9}{c:>8}{mark}")

    print("\nrep3: lower=better | dist2/coh: higher=better")
    print("Baseline k=0 = fixed optimal temperature (0.8).")


if __name__ == "__main__":
    main()
