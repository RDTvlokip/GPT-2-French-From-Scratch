"""Evaluation harness: chiffre la qualité d'un modèle, de façon comparable
entre runs.

Métriques produites :
- val_ppl          : perplexité sur le set de validation (data/val.pt).
- repetition_3gram : % de 3-grammes répétés dans le texte généré (radotage).
- distinct_2       : bigrammes uniques / total (diversité lexicale, 0..1).
- coherence_len    : longueur moyenne (en tokens) générée avant un signe de
                     dérive (token '#' = nouvelle section, ou boucle de
                     répétition). Proxy automatique de "tient un sujet".

Prompts d'évaluation :
- auto   : débuts de blocs tirés de val.pt (texte non vu en entraînement),
- fixe   : un petit set écrit à la main (scripts/eval_prompts.txt si présent),
           sinon un set par défaut intégré.

Sortie : tableau console + une ligne dans logs/evaluations.csv + un JSON
horodaté dans logs/evaluations/.

Usage:
    python scripts/evaluate.py --model models/best_model.pt
    python scripts/evaluate.py --model logs/<run>/best_model.pt --n-prompts 30
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

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

DEFAULT_PROMPTS = [
    "La ville de Paris est",
    "L'intelligence artificielle est",
    "Le protocole TCP permet",
    "Pendant la Seconde Guerre mondiale,",
    "La photosynthèse est un processus",
]


# --------------------------------------------------------------------------- #
# Métriques de texte (sur des listes de token ids générés)
# --------------------------------------------------------------------------- #
def repetition_3gram(token_ids):
    """Fraction de 3-grammes qui sont des répétitions (1 - unique/total)."""
    if len(token_ids) < 3:
        return 0.0
    grams = [tuple(token_ids[i:i + 3]) for i in range(len(token_ids) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def distinct_2(token_ids):
    """Bigrammes distincts / total bigrammes (diversité, plus haut = mieux)."""
    if len(token_ids) < 2:
        return 0.0
    grams = [tuple(token_ids[i:i + 2]) for i in range(len(token_ids) - 1)]
    return len(set(grams)) / len(grams)


def coherence_len(token_ids, drift_tokens):
    """Nb de tokens générés avant le premier signe de dérive.

    Dérive = apparition d'un token de "nouvelle section" (drift_tokens, ex '#'),
    OU le début d'une boucle (un 3-gramme déjà vu se répète). Capé à len.
    """
    seen = set()
    for i, tid in enumerate(token_ids):
        if tid in drift_tokens:
            return i
        if i >= 2:
            g = (token_ids[i - 2], token_ids[i - 1], tid)
            if g in seen:
                return i
            seen.add(g)
    return len(token_ids)


# --------------------------------------------------------------------------- #
# Perplexité sur le set de validation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_val_ppl(model, val_path, device, max_blocks=200, batch_size=8):
    """PPL moyenne sur un échantillon du set de validation."""
    if not os.path.exists(val_path):
        return None
    data = torch.load(val_path, weights_only=True)
    n = min(max_blocks, len(data))
    data = data[:n]
    total_loss, total_count = 0.0, 0
    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size].to(device)
        _, loss, _, _ = model(batch, labels=batch)
        # loss est déjà la moyenne du batch ; on pondère par le nb de blocs
        bs = batch.size(0)
        total_loss += loss.item() * bs
        total_count += bs
    mean_loss = total_loss / max(total_count, 1)
    return float(torch.exp(torch.tensor(mean_loss)))


# --------------------------------------------------------------------------- #
# Génération simple (greedy-ish) pour les métriques de texte
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_ids(model, tokenizer, prompt_ids, device, max_new=120,
                 temperature=0.7, top_k=30, ban_bos=True, greedy=True):
    """Génère max_new tokens à partir de prompt_ids. Retourne les NOUVEAUX ids.

    greedy=True (défaut pour l'éval) : argmax déterministe -> mêmes résultats à
    chaque exécution, indispensable pour COMPARER deux modèles sans bruit
    d'échantillonnage. greedy=False : sampling temperature/top_k.
    """
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    n_positions = model.config.n_positions
    generated = []
    for _ in range(max_new):
        cond = ids if ids.size(1) <= n_positions else ids[:, -n_positions:]
        logits = model(cond)[0][:, -1, :]
        if ban_bos:
            logits[:, tokenizer.bos_token_id] = float("-inf")
        if greedy:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[logits < kth] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tid = nxt.item()
        if tid == tokenizer.eos_token_id:
            break
        generated.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return generated


# --------------------------------------------------------------------------- #
# Construction des prompts
# --------------------------------------------------------------------------- #
def build_prompts(tokenizer, val_path, n_auto, prompt_len=12):
    """Retourne (auto_prompts_ids, fixed_prompts_ids)."""
    auto = []
    if os.path.exists(val_path):
        data = torch.load(val_path, weights_only=True)
        # tire n_auto blocs régulièrement espacés, prend leurs prompt_len 1ers
        step = max(1, len(data) // max(n_auto, 1))
        for i in range(0, len(data), step):
            block = data[i].tolist()
            # saute le <bos> initial s'il y est, garde prompt_len tokens
            start = 1 if block and block[0] == tokenizer.bos_token_id else 0
            auto.append(block[start:start + prompt_len])
            if len(auto) >= n_auto:
                break

    fixed_file = PROJECT_ROOT / "scripts" / "eval_prompts.txt"
    if fixed_file.exists():
        texts = [l.strip() for l in fixed_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        texts = DEFAULT_PROMPTS
    fixed = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=True)
        if ids and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]  # ne pas signaler une fin de doc (cf bug #6)
        fixed.append(ids)
    return auto, fixed


# --------------------------------------------------------------------------- #
def evaluate(model, tokenizer, prompts, device, drift_tokens, max_new=120, greedy=False):
    """Moyenne les métriques de texte sur une liste de prompts (ids)."""
    reps, dists, cohs = [], [], []
    for pids in prompts:
        gen = generate_ids(model, tokenizer, pids, device, max_new=max_new, greedy=greedy)
        if not gen:
            continue
        reps.append(repetition_3gram(gen))
        dists.append(distinct_2(gen))
        cohs.append(coherence_len(gen, drift_tokens))
    nz = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "repetition_3gram": round(nz(reps), 4),
        "distinct_2": round(nz(dists), 4),
        "coherence_len": round(nz(cohs), 1),
        "n_prompts": len(reps),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate a GPT-2 checkpoint")
    ap.add_argument("--model", default="models/best_model.pt", help="checkpoint path")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "bpe_tokenizer_32k.json"))
    ap.add_argument("--val", default=str(PROJECT_ROOT / "data" / "val.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-prompts", type=int, default=20, help="nb de prompts auto")
    ap.add_argument("--max-new", type=int, default=120, help="tokens générés/prompt")
    ap.add_argument("--greedy", action="store_true",
                    help="Greedy decoding (déterministe mais amplifie les "
                         "répétitions). Défaut : sampling à seed fixe "
                         "(reproductible ET réaliste).")
    ap.add_argument("--seed", type=int, default=42, help="seed du sampling")
    args = ap.parse_args()

    # Seed fixe -> sampling reproductible (mêmes tirages à chaque exécution),
    # donc deux modèles sont comparables sans bruit d'échantillonnage.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = args.model if os.path.isabs(args.model) else str(PROJECT_ROOT / args.model)
    tokenizer = GPT2Tokenizer(args.tokenizer)
    model = GPT2.from_pretrained(model_path, device=args.device)
    model.eval()

    # tokens qui signalent une "nouvelle section" (dérive). '#' -> id 957, '##' -> 552
    drift = set()
    for s in ["#", " #", "##", " ##"]:
        for tid in tokenizer.encode(s, add_special_tokens=False):
            drift.add(tid)

    print("Building prompts...")
    auto, fixed = build_prompts(tokenizer, args.val, args.n_prompts)

    print("Computing validation perplexity...")
    val_ppl = compute_val_ppl(model, args.val, args.device)

    print(f"Evaluating on {len(auto)} auto + {len(fixed)} fixed prompts...")
    m_auto = evaluate(model, tokenizer, auto, args.device, drift, args.max_new, args.greedy)
    m_fixed = evaluate(model, tokenizer, fixed, args.device, drift, args.max_new, args.greedy)

    result = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "model": model_path,
        "params_M": round(model.get_num_params() / 1e6, 2),
        "val_ppl": round(val_ppl, 3) if val_ppl is not None else None,
        "auto": m_auto,
        "fixed": m_fixed,
    }

    # ---- Affichage console ----
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Model:         {model_path}")
    print(f"Params:        {result['params_M']}M")
    print(f"Val PPL:       {result['val_ppl']}")
    print(f"{'metric':<20}{'auto':>12}{'fixed':>12}")
    print("-" * 44)
    for k in ["repetition_3gram", "distinct_2", "coherence_len"]:
        print(f"{k:<20}{m_auto[k]:>12}{m_fixed[k]:>12}")
    print("=" * 60)
    print("repetition_3gram: lower=better | distinct_2: higher=better | "
          "coherence_len: higher=better")

    # ---- CSV (historique) ----
    # Each eval gets its OWN dated folder (like a training run), holding its
    # metrics.csv + result.json. Self-contained, no shared file to pollute.
    model_tag = Path(model_path).parent.name or Path(model_path).stem
    decoding = "greedy" if args.greedy else f"sample_seed{args.seed}"
    eval_dir = PROJECT_ROOT / "logs" / "evaluations" / f"{result['timestamp']}_{model_tag}_{decoding}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("timestamp,model,params_M,val_ppl,decoding,"
                "auto_rep3,auto_distinct2,auto_coh,"
                "fixed_rep3,fixed_distinct2,fixed_coh\n")
        f.write(f"{result['timestamp']},{model_path},{result['params_M']},{result['val_ppl']},{decoding},"
                f"{m_auto['repetition_3gram']},{m_auto['distinct_2']},{m_auto['coherence_len']},"
                f"{m_fixed['repetition_3gram']},{m_fixed['distinct_2']},{m_fixed['coherence_len']}\n")

    result["decoding"] = decoding
    json_path = eval_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nEval folder  {eval_dir}")
    print(f"  metrics.csv + result.json")


if __name__ == "__main__":
    main()
