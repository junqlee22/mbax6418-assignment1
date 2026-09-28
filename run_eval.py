"""Step 2: classify first 100 Gift Cards reviews and report metrics.

Pipeline:
 1. Load first 100 rows from Gift_Cards.jsonl.gz
 2. Extract title, text, rating, asin, user_id
 3. classify_review(title, text) — rating NEVER shown to the model
 4. Ground truth from rating AFTER classification: >=4 POSITIVE else NEGATIVE
 5. Save results_100.csv
 6. Report overall + per-class accuracy, misclassifications, class skew
"""

import csv
import gzip
import json

from review_classifier import classify_review

SRC = "Gift_Cards.jsonl.gz"
OUT = "results_100.csv"
N = 100


def load_rows(path: str, n: int) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows


def ground_truth(rating: float) -> str:
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def main() -> None:
    rows = load_rows(SRC, N)

    results = []
    for i, r in enumerate(rows):
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        rating = r.get("rating")
        gt = ground_truth(rating)
        # Model sees ONLY title+text; rating is excluded from the prompt.
        predicted = classify_review(title, text)
        results.append(
            {
                "index": i,
                "title": title,
                "text": text,
                "rating": rating,
                "ground_truth": gt,
                "predicted": predicted,
                "correct": gt == predicted,
            }
        )

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "title",
                "text",
                "rating",
                "ground_truth",
                "predicted",
                "correct",
            ],
        )
        w.writeheader()
        w.writerows(results)

    total = len(results)
    acc = sum(r["correct"] for r in results) / total * 100

    # Class distribution (ground truth)
    from collections import Counter

    gt_counts = Counter(r["ground_truth"] for r in results)
    pos_total = gt_counts["POSITIVE"]
    neg_total = gt_counts["NEGATIVE"]

    pos_correct = sum(
        r["correct"] for r in results if r["ground_truth"] == "POSITIVE"
    )
    neg_correct = sum(
        r["correct"] for r in results if r["ground_truth"] == "NEGATIVE"
    )
    pos_acc = pos_correct / pos_total * 100 if pos_total else 0.0
    neg_acc = neg_correct / neg_total * 100 if neg_total else 0.0

    mis = [r for r in results if not r["correct"]]

    print("=" * 70)
    print(f"Rows classified: {total}")
    print(f"Class distribution (ground truth): "
          f"POSITIVE {pos_total} ({pos_total/total*100:.1f}%), "
          f"NEGATIVE {neg_total} ({neg_total/total*100:.1f}%)")
    print("=" * 70)
    print(f"Overall accuracy: {acc:.2f}% ({sum(r['correct'] for r in results)}/{total})")
    print(f"Per-class accuracy:")
    print(f"  POSITIVE as POSITIVE: {pos_correct}/{pos_total} = {pos_acc:.2f}%")
    print(f"  NEGATIVE as NEGATIVE: {neg_correct}/{neg_total} = {neg_acc:.2f}%")
    print(f"Misclassified: {len(mis)}")
    print("-" * 70)
    for r in mis:
        print(f"  [{r['index']}] title={r['title'][:40]!r} | star={r['rating']} "
              f"| gt={r['ground_truth']} | pred={r['predicted']}")
    print("=" * 70)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
