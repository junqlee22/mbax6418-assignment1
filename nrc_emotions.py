"""Step 5 (take 2): derive each review's primary emotion from the NRC lexicon.

Word-list method (no model calls): scores every review's tokens against the
NRC Emotion Lexicon (word-level, v0.92) -- a public list mapping ~14k English
words to one or more of eight emotions (anger, anticipation, disgust, fear,
joy, sadness, surprise, trust). Per review we sum the 0/1 association flags
per emotion over the review's tokens; the emotion with the highest total is
the review's derived primary emotion. Ties are broken by the canonical
emotion order. Reviews with no lexicon hits get no emotion (None) and a
coverage of 0.

Usage:
    python nrc_emotions.py

Reads results_balanced.csv (LLM 3-class + LLM emotion) and results_100.csv,
adds the derived emotion columns, and writes:
    outputs/emotions_balanced.csv
    outputs/emotions_100.csv
"""

import csv
import html
import os
import re
from collections import Counter

EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
LEXICON_PATH = "data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

_BALANCED_IN = "results_balanced.csv"
_BALANCED_OUT = "outputs/emotions_balanced.csv"
_FIRST100_IN = "results_100.csv"
_FIRST100_OUT = "outputs/emotions_100.csv"

_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z']+")


def load_lexicon(path: str) -> dict[str, set[str]]:
    """word -> set of associated emotions (flag==1 only)."""
    lexicon: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if emotion not in EMOTIONS or flag != "1":
                continue
            lexicon.setdefault(word, set()).add(emotion)
    return lexicon


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; strip HTML tags/entities so markup never scores."""
    if not text:
        return []
    cleaned = html.unescape(_TAG_RE.sub(" ", text or ""))
    return _TOKEN_RE.findall(cleaned.lower())


def score_text(text: str, lexicon: dict[str, set[str]]) -> tuple[str | None, dict, int]:
    """Return (primary emotion, per-emotion scores, number of hit tokens)."""
    scores = {e: 0 for e in EMOTIONS}
    hit_tokens = 0
    for token in tokenize(text):
        emos = lexicon.get(token)
        if emos:
            hit_tokens += 1
            for e in emos:
                scores[e] += 1
    total = sum(scores.values())
    if total == 0:
        return None, scores, hit_tokens
    # tie-break: canonical emotion order
    primary = max(EMOTIONS, key=lambda e: (scores[e], -EMOTIONS.index(e)))
    return primary, scores, hit_tokens


def process(path_in: str, path_out: str, lexicon: dict[str, set[str]]) -> None:
    rows = list(csv.DictReader(open(path_in, encoding="utf-8")))
    coverage_hits, coverage_rows = 0, 0
    out_rows = []
    for r in rows:
        text = f"{r.get('title', '')} {r.get('text', '')}".strip()
        primary, scores, hits = score_text(text, lexicon)
        if hits:
            coverage_rows += 1
        coverage_hits += hits
        row = {**r, "emotion_nrc": primary or "", "nrc_hit_tokens": hits}
        for e in EMOTIONS:
            row[f"nrc_{e}"] = scores[e]
        out_rows.append(row)
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)
    coverage = coverage_rows / len(rows) * 100 if rows else 0.0
    print(f"{path_in} -> {path_out}")
    print(f"  rows={len(rows)}  rows_with_any_hit={coverage_rows} ({coverage:.1f}%)  "
          f"avg hit tokens/review={coverage_hits / len(rows):.2f}")


def main() -> None:
    if not os.path.exists(LEXICON_PATH):
        raise SystemExit(
            f"lexicon not found at {LEXICON_PATH} -- download NRC-Emotion-Lexicon.zip "
            "from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm and extract "
            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt into data/"
        )
    lexicon = load_lexicon(LEXICON_PATH)
    print(f"lexicon: {len(lexicon)} words, {sum(len(v) for v in lexicon.values())} word-emotion pairs")
    for e in EMOTIONS:
        print(f"  {e:14} {sum(1 for v in lexicon.values() if e in v)} words")
    print("-" * 60)
    process(_BALANCED_IN, _BALANCED_OUT, lexicon)
    process(_FIRST100_IN, _FIRST100_OUT, lexicon)


if __name__ == "__main__":
    main()
