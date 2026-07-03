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

    NB : métrique GAMEABLE — un décodage peut gonfler ce score en évitant les
    '#' tout en produisant du texte décousu. Croiser avec les métriques
    anti-gaming ci-dessous (self_ppl, proper_noun_ratio, prompt_overlap).
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


# --- Métriques anti-gaming (texte décodé) ------------------------------------ #
import re as _re

_WORD_RE = _re.compile(r"\b\w[\w'-]*\b", _re.UNICODE)
# mots-outils français à ignorer pour l'overlap de contenu
_STOP = set("le la les un une des de du d l et à a en au aux ce cette ces son sa "
            "ses qui que dont pour par sur dans est sont être avoir il elle ils "
            "elles on se sa ou où mais donc or ni car plus very".split())


def proper_noun_ratio(text):
    """Fraction de mots Capitalisés en MILIEU de phrase (proxy noms inventés).

    Un texte qui hallucine des noms ('Nashville (Ontario)', 'James Scott
    Eiffles-Clermont') en produit beaucoup. On ignore le 1er mot de chaque
    phrase (capitalisé par convention).
    """
    sentences = _re.split(r"[.!?]\s+", text)
    mid_caps, total = 0, 0
    for s in sentences:
        words = _WORD_RE.findall(s)
        for j, w in enumerate(words):
            if j == 0:
                continue  # début de phrase : capitale normale
            total += 1
            if w[0].isupper():
                mid_caps += 1
    return round(mid_caps / total, 4) if total else 0.0


def prompt_overlap(prompt_text, gen_text):
    """Overlap de mots-contenu entre prompt et génération (cohérence lexicale).

    Plus haut = la génération reste lexicalement liée au prompt (moins de drift
    sémantique). 0 = aucun mot-contenu du prompt réutilisé.
    """
    def content(t):
        return {w.lower() for w in _WORD_RE.findall(t)
                if w.lower() not in _STOP and len(w) > 2}
    p, g = content(prompt_text), content(gen_text)
    if not p:
        return 0.0
    return round(len(p & g) / len(p), 4)


def sentence_burstiness(text):
    """Coefficient de variation des longueurs de phrases (rythme).

    Texte fluide = longueurs régulières (CV bas). Texte décousu = CV haut.
    Plus BAS = mieux.
    """
    lens = [len(_WORD_RE.findall(s)) for s in _re.split(r"[.!?]\s+", text) if s.strip()]
    lens = [n for n in lens if n > 0]
    if len(lens) < 2:
        return 0.0
    mean = sum(lens) / len(lens)
    var = sum((n - mean) ** 2 for n in lens) / len(lens)
    return round((var ** 0.5) / mean, 4) if mean else 0.0


@torch.no_grad()
def self_perplexity(model, token_ids, device):
    """Le modèle re-score son propre texte généré. Plus BAS = plus cohérent.

    Un texte décousu/halluciné a une perplexité interne plus haute, même s'il
    évite les '#'. C'est la métrique anti-gaming la plus robuste.
    """
    if len(token_ids) < 2:
        return None
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    _, loss, _, _ = model(ids, labels=ids)
    return float(torch.exp(loss))


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
    """Moyenne les métriques de texte sur une liste de prompts (ids).

    Inclut les métriques anti-gaming : self_ppl, proper_noun_ratio,
    prompt_overlap, burstiness — pour détecter un texte décousu/halluciné que
    coherence_len ne voit pas.
    """
    reps, dists, cohs = [], [], []
    sppls, pnrs, ovls, bursts = [], [], [], []
    for pids in prompts:
        gen = generate_ids(model, tokenizer, pids, device, max_new=max_new, greedy=greedy)
        if not gen:
            continue
        reps.append(repetition_3gram(gen))
        dists.append(distinct_2(gen))
        cohs.append(coherence_len(gen, drift_tokens))
        # anti-gaming (sur texte décodé + self-PPL)
        gen_text = tokenizer.decode(gen, skip_special_tokens=True)
        prompt_text = tokenizer.decode(pids, skip_special_tokens=True)
        pnrs.append(proper_noun_ratio(gen_text))
        ovls.append(prompt_overlap(prompt_text, gen_text))
        bursts.append(sentence_burstiness(gen_text))
        sp = self_perplexity(model, gen, device)
        if sp is not None and sp < 1e6:  # garde-fou valeurs aberrantes
            sppls.append(sp)
    nz = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "repetition_3gram": round(nz(reps), 4),
        "distinct_2": round(nz(dists), 4),
        "coherence_len": round(nz(cohs), 1),
        "self_ppl": round(nz(sppls), 2),
        "proper_noun_ratio": round(nz(pnrs), 4),
        "prompt_overlap": round(nz(ovls), 4),
        "burstiness": round(nz(bursts), 4),
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
    ap.add_argument("--adaptive-halting", action="store_true",
                    help="Active le halting adaptatif par token (looped only): "
                         "les tokens confiants sortent tôt de la boucle R.")
    ap.add_argument("--halting-threshold", type=float, default=1.0,
                    help="Seuil d'entropie (nats) pour figer un token (mode absolute).")
    ap.add_argument("--halting-mode", choices=["absolute", "percentile"], default=None,
                    help="Mode de halting à l'inférence (défaut: celui du checkpoint).")
    ap.add_argument("--halting-percentile", type=float, default=None,
                    help="q: fraction de tokens actifs à figer/itération (mode percentile).")
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

    # Optional: enable adaptive per-token halting after load (inference-time).
    # IMPORTANT: the inference halting mode must match how the model was TRAINED
    # (train/inference coherence — see option-A failure). By default we keep the
    # checkpoint's own config; flags below only override if explicitly passed.
    if args.adaptive_halting:
        model.config.adaptive_halting = True
    if args.halting_mode is not None:
        model.config.halting_mode = args.halting_mode
    if args.halting_percentile is not None:
        model.config.halting_percentile = args.halting_percentile
    if getattr(args, "halting_threshold", None) is not None and args.adaptive_halting:
        model.config.halting_entropy_threshold = args.halting_threshold
    if getattr(model.config, "adaptive_halting", False):
        mode = getattr(model.config, "halting_mode", "absolute")
        detail = (f"percentile q={model.config.halting_percentile}" if mode == "percentile"
                  else f"absolute thr={model.config.halting_entropy_threshold}")
        print(f"Adaptive halting ON ({detail})")

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
    metric_keys = ["repetition_3gram", "distinct_2", "coherence_len",
                   "self_ppl", "proper_noun_ratio", "prompt_overlap", "burstiness"]
    print(f"{'metric':<20}{'auto':>12}{'fixed':>12}")
    print("-" * 44)
    for k in metric_keys:
        print(f"{k:<20}{m_auto[k]:>12}{m_fixed[k]:>12}")
    print("=" * 60)
    print("lower=better : rep3, self_ppl, proper_noun_ratio, burstiness")
    print("higher=better: distinct_2, coherence_len, prompt_overlap")
    print("(self_ppl / proper_noun_ratio = anti-gaming : détectent le texte")
    print(" décousu/halluciné que coherence_len ne voit pas)")

    # ---- CSV (historique) ----
    # Each eval gets its OWN dated folder (like a training run), holding its
    # metrics.csv + result.json. Self-contained, no shared file to pollute.
    model_tag = Path(model_path).parent.name or Path(model_path).stem
    decoding = "greedy" if args.greedy else f"sample_seed{args.seed}"
    eval_dir = PROJECT_ROOT / "logs" / "evaluations" / f"{result['timestamp']}_{model_tag}_{decoding}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    def row(m):
        return ",".join(str(m[k]) for k in metric_keys)

    csv_path = eval_dir / "metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        cols = ",".join(metric_keys)
        f.write(f"timestamp,model,params_M,val_ppl,decoding,split,{cols}\n")
        f.write(f"{result['timestamp']},{model_path},{result['params_M']},{result['val_ppl']},{decoding},auto,{row(m_auto)}\n")
        f.write(f"{result['timestamp']},{model_path},{result['params_M']},{result['val_ppl']},{decoding},fixed,{row(m_fixed)}\n")

    result["decoding"] = decoding
    json_path = eval_dir / "result.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nEval folder  {eval_dir}")
    print(f"  metrics.csv + result.json")


if __name__ == "__main__":
    main()
