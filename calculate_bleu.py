import re
import argparse
from typing import Optional

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
    _SMOOTH = SmoothingFunction().method3
except Exception:
    _HAS_NLTK = False
    _SMOOTH = None


# -------------------- Text helpers --------------------

def _normalize(s: Optional[str]) -> str:
    """Lowercase, trim, collapse whitespace, and strip outer quotes if present."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1]
    return s


# -------------------- Metric --------------------

def _bleu(ref: Optional[str], hyp: Optional[str]) -> float:
    """
    Sentence-level BLEU with smoothing (if nltk is available).
    Fallback: simple token overlap ratio if nltk is unavailable.
    Edge cases:
      - both empty -> 1.0
      - one empty  -> 0.0
    """
    ref_norm = _normalize(ref)
    hyp_norm = _normalize(hyp)

    ref_empty = (ref_norm == "")
    hyp_empty = (hyp_norm == "")

    if ref_empty and hyp_empty:
        return 1.0
    if ref_empty != hyp_empty:
        return 0.0

    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()

    if not ref_tokens or not hyp_tokens:
        return 0.0

    if _HAS_NLTK:
        return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH))

    # Fallback overlap ratio
    if ref_norm == hyp_norm:
        return 1.0
    common = sum(min(ref_tokens.count(t), hyp_tokens.count(t)) for t in set(hyp_tokens))
    return common / max(len(hyp_tokens), 1)


def compute_bleu(gold: str, reason: str) -> float:
    """Public helper: compute BLEU between a gold label and a predicted reason."""
    return _bleu(gold, reason)


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute BLEU for a single (gold, reason) pair.")
    p.add_argument("--gold", "--ref", dest="gold", required=True, help="Gold label text (reference).")
    p.add_argument("--reason", "--pred", "--hyp", dest="reason", required=True, help="Target reason text (hypothesis).")
    p.add_argument("--verbose", action="store_true", help="Print normalized texts and tokenization info.")
    return p.parse_args()


def main():
    args = parse_args()
    score = compute_bleu(args.gold, args.reason)
    if args.verbose:
        print(f"[gold(norm) ]: {_normalize(args.gold)}")
        print(f"[reason(norm)]: {_normalize(args.reason)}")
        print(f"[BLEU]: {score:.6f}")
    else:
        # Print BLEU only for easy piping
        print(f"{score:.6f}")


if __name__ == "__main__":
    main()
