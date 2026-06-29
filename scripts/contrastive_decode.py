"""Contrastive Decoding (Li et al. 2022) entre deux checkpoints.

Variante testée ici : expert et amateur ont la MÊME taille (15M) mais un
degré d'entraînement différent (ex: 60%×6 epochs = expert, 20%×3 = amateur).
On teste si contraster sur le *degré d'entraînement* (à taille égale) améliore
la génération — inédit à cette échelle sur du français.

Méthode (fidèle au papier) :
- plausibility constraint (α-masking) : on ne garde que les tokens où l'expert
  est plausible,  V_valid = { t : p_expert(t) >= α * max_t' p_expert(t') }.
  Sans ça, le contraste amplifie des tokens rares aberrants.
- score contrastif sur V_valid :
      logits_cd = (1 + λ) * logits_expert - λ * logits_amateur
  (équivalent au log-ratio expert/amateur à une constante près, sur V_valid).

Évaluation : balayage (λ, α), métriques rep3 / distinct2 / coherence du
harnais evaluate.py, comparées à la baseline (expert seul, λ=0).

Usage:
    python scripts/contrastive_decode.py \
        --expert models/best_model.pt \
        --amateur logs/<run_20pct_3ep>/best_model.pt
"""

import sys
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
# Réutilise les métriques + la construction de prompts du harnais d'éval.
from scripts.evaluate import (
    repetition_3gram, distinct_2, coherence_len, build_prompts,
)


@torch.no_grad()
def cd_generate(expert, amateur, tokenizer, prompt_ids, device,
                max_new=120, lam=1.0, alpha=0.1, temperature=0.8,
                top_k=40, rep_pen=1.3, ban_bos=True):
    """Génère avec contrastive decoding. lam=0 -> expert seul (baseline).

    IMPORTANT : on applique le MÊME post-traitement de sampling (rep_pen,
    top_k) que la config optimale, à la baseline ET au CD, pour une
    comparaison juste (sinon on compare deux générations non optimisées).
    """
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_pos = expert.config.n_positions
    generated = []
    for _ in range(max_new):
        cond = ids if ids.size(1) <= n_pos else ids[:, -n_pos:]
        log_e = expert(cond)[0][:, -1, :]
        if ban_bos:
            log_e[:, tokenizer.bos_token_id] = float("-inf")

        if lam == 0.0:
            scores = log_e  # baseline : expert seul
        else:
            log_a = amateur(cond)[0][:, -1, :]
            # Plausibility constraint : masque hors V_valid (p_e < alpha*max).
            probs_e = F.softmax(log_e, dim=-1)
            thresh = alpha * probs_e.max(dim=-1, keepdim=True).values
            valid = probs_e >= thresh
            scores = (1.0 + lam) * log_e - lam * log_a
            scores = scores.masked_fill(~valid, float("-inf"))

        # Sampling identique à la config optimale (équité baseline vs CD).
        if rep_pen != 1.0:
            for t in set(ids[0].tolist()):
                v = scores[0, t]
                scores[0, t] = v / rep_pen if v > 0 else v * rep_pen
        scores = scores / temperature
        if top_k:
            kth = torch.topk(scores, min(top_k, scores.size(-1)))[0][..., -1, None]
            scores[scores < kth] = float("-inf")
        probs = F.softmax(scores, dim=-1)
        nxt = torch.multinomial(probs, 1)
        tid = nxt.item()
        if tid == tokenizer.eos_token_id:
            break
        generated.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return generated


def eval_config(expert, amateur, tokenizer, prompts, device, drift,
                lam, alpha, max_new, seed=42):
    """Métriques moyennes pour un réglage (lam, alpha)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reps, dists, cohs = [], [], []
    for pids in prompts:
        gen = cd_generate(expert, amateur, tokenizer, pids, device,
                          max_new=max_new, lam=lam, alpha=alpha)
        if not gen:
            continue
        reps.append(repetition_3gram(gen))
        dists.append(distinct_2(gen))
        cohs.append(coherence_len(gen, drift))
    nz = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return round(nz(reps), 4), round(nz(dists), 4), round(nz(cohs), 1)


def main():
    ap = argparse.ArgumentParser(description="Contrastive decoding between checkpoints")
    ap.add_argument("--expert", default="models/best_model.pt")
    ap.add_argument("--amateur", required=True, help="checkpoint moins entraîné")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "bpe_tokenizer_32k.json"))
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data" / "val.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--lambdas", default="0.5,1.0", help="valeurs de λ à tester")
    ap.add_argument("--alphas", default="0.1,0.3", help="valeurs de α à tester")
    args = ap.parse_args()

    def resolve(p):
        return p if Path(p).is_absolute() else str(PROJECT_ROOT / p)

    tokenizer = GPT2Tokenizer(args.tokenizer)
    print("Loading expert...")
    expert = GPT2.from_pretrained(resolve(args.expert), device=args.device); expert.eval()
    print("Loading amateur...")
    amateur = GPT2.from_pretrained(resolve(args.amateur), device=args.device); amateur.eval()

    drift = set()
    for s in ["#", " #", "##", " ##"]:
        for tid in tokenizer.encode(s, add_special_tokens=False):
            drift.add(tid)

    auto, fixed = build_prompts(tokenizer, args.val, args.n_prompts)
    prompts = auto + fixed
    print(f"Evaluating on {len(prompts)} prompts "
          f"({len(auto)} auto + {len(fixed)} fixed)\n")

    lambdas = [float(x) for x in args.lambdas.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"{'config':<22}{'rep3↓':>9}{'dist2↑':>9}{'coh↑':>8}")
    print("-" * 48)
    # Baseline : expert seul (lam=0)
    r, d, c = eval_config(expert, amateur, tokenizer, prompts, args.device,
                          drift, 0.0, 0.0, args.max_new)
    print(f"{'baseline (expert)':<22}{r:>9}{d:>9}{c:>8}")
    base = (r, d, c)

    for lam in lambdas:
        for alpha in alphas:
            r, d, c = eval_config(expert, amateur, tokenizer, prompts,
                                  args.device, drift, lam, alpha, args.max_new)
            tag = f"CD λ={lam} α={alpha}"
            # flèches d'amélioration vs baseline
            mark = ""
            if r < base[0]:
                mark += " rep↓"
            if d > base[1]:
                mark += " div↑"
            if c > base[2]:
                mark += " coh↑"
            print(f"{tag:<22}{r:>9}{d:>9}{c:>8}{mark}")

    print("\nrep3: lower=better | dist2/coh: higher=better")
    print("Baseline = expert seul (pas de contraste).")


if __name__ == "__main__":
    main()
