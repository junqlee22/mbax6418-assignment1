"""Step 6: balanced three-class run with LLM sentiment + primary emotion.

Pipeline:
 1. Load every review from Gift_Cards.jsonl.gz (whole file).
 2. Ground-truth classes: rating 4-5 -> POSITIVE, 3 -> NEUTRAL, 1-2 -> NEGATIVE.
 3. Pull a balanced sample: ~PER_CLASS reviews per class, chosen with a FIXED
    random seed so the same set comes up every run.
 4. Classify each with classify_full() (3-class sentiment + one of 8 emotions).
    The rating is NEVER sent to the model -- only title + text.
 5. Save outputs/balanced_raw.jsonl (raw API responses) and
    results_balanced.csv (parsed results), plus outputs/dataset_stats.json
    (whole-file rating/class distribution used by the dashboard).
 6. Print confusion matrix and per-class metrics.

Resumable: sample ids already present in the raw output file are skipped.
"""

import argparse
import csv
import gzip
import json
import os
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from review_classifier import classify_full

SRC = "Gift_Cards.jsonl.gz"
RAW_OUT = "outputs/balanced_raw.jsonl"
CSV_OUT = "results_balanced.csv"
STATS_OUT = "outputs/dataset_stats.json"
DEFAULT_SEED = 6418
DEFAULT_PER_CLASS = 50
DEFAULT_WORKERS = 4
RETRIES = 3


def three_class(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def load_all_reviews(path: str) -> list[dict]:
    reviews = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            reviews.append(
                {
                    "file_index": len(reviews),
                    "title": (r.get("title") or "").strip(),
                    "text": (r.get("text") or "").strip(),
                    "rating": float(r.get("rating")),
                    "asin": r.get("asin", ""),
                    "parent_asin": r.get("parent_asin", ""),
                    "user_id": r.get("user_id", ""),
                    "verified_purchase": bool(r.get("verified_purchase")),
                    "helpful_vote": int(r.get("helpful_vote") or 0),
                    "timestamp": r.get("timestamp", 0),
                }
            )
    return reviews


def pick_balanced_sample(reviews: list[dict], per_class: int, seed: int) -> list[dict]:
    by_class: dict[str, list[dict]] = {"POSITIVE": [], "NEUTRAL": [], "NEGATIVE": []}
    for r in reviews:
        by_class[three_class(r["rating"])].append(r)
    rng = random.Random(seed)
    sample: list[dict] = []
    for cls, members in by_class.items():
        k = min(per_class, len(members))
        chosen = rng.sample(members, k)
        for r in chosen:
            r["true_class"] = cls
        sample.extend(chosen)
        print(f"{cls:9} available={len(members):6}  sampled={k}")
    rng.shuffle(sample)  # interleave classes so a crash doesn't bias one class
    return sample


def classify_one(review: dict, timeout: float = 90.0) -> dict:
    """Classify one review; returns dict with prediction fields or error."""
    last_err = None
    for attempt in range(RETRIES):
        try:
            result = classify_full(review["title"], review["text"], timeout=timeout)
            return {"predicted": result["sentiment"], "emotion_llm": result["emotion"]}
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    return {"predicted": "", "emotion_llm": "", "error": repr(last_err)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    reviews = load_all_reviews(SRC)
    print(f"loaded {len(reviews):,} reviews from {SRC}")

    os.makedirs("outputs", exist_ok=True)
    rating_counts = Counter(r["rating"] for r in reviews)
    class_counts = Counter(three_class(r["rating"]) for r in reviews)
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_reviews": len(reviews),
                "seed": args.seed,
                "per_class_target": args.per_class,
                "rating_counts": {str(k): v for k, v in sorted(rating_counts.items())},
                "class_counts": dict(class_counts),
            },
            f,
            indent=2,
        )
    print("dataset stats ->", STATS_OUT)

    sample = pick_balanced_sample(reviews, args.per_class, args.seed)
    print(f"classified sample size: {len(sample)}")

    # Resume: sample ids already in the raw file.
    done_ids: set[int] = set()
    try:
        with open(RAW_OUT, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["file_index"])
    except FileNotFoundError:
        pass
    todo = [r for r in sample if r["file_index"] not in done_ids]
    print(f"already done: {len(sample) - len(todo)}  to classify: {len(todo)}")

    lock = threading.Lock()
    finished = 0

    def handle(review: dict) -> dict:
        nonlocal finished
        pred = classify_one(review)
        with lock:
            with open(RAW_OUT, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"file_index": review["file_index"], "prediction": pred},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            finished += 1
            if finished % 10 == 0 or finished == len(todo):
                print(f"  progress {finished}/{len(todo)}")
        return {**review, **pred}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(handle, r) for r in todo]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print("worker failure:", e)

    # Reconcile: rows already classified (from resume) + fresh results.
    fresh = {r["file_index"]: r for r in results}
    all_rows = []
    for r in sample:
        row = fresh.get(r["file_index"])
        if row is None:
            # previously completed: read back from raw file
            pred = {"predicted": "", "emotion_llm": "", "error": "missing"}
        else:
            pred = row
        all_rows.append(
            {
                "file_index": r["file_index"],
                "title": r["title"],
                "text": r["text"],
                "rating": r["rating"],
                "ground_truth": r["true_class"],
                "predicted": pred.get("predicted", ""),
                "emotion_llm": pred.get("emotion_llm", ""),
                "error": pred.get("error", ""),
                "asin": r["asin"],
                "verified_purchase": r["verified_purchase"],
                "helpful_vote": r["helpful_vote"],
                "correct": pred.get("predicted", "") == r["true_class"],
            }
        )

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file_index", "title", "text", "rating", "ground_truth",
                "predicted", "emotion_llm", "error", "asin",
                "verified_purchase", "helpful_vote", "correct",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved -> {CSV_OUT}")

    # --- Summary metrics ---
    ok = [r for r in all_rows if not r["error"]]
    n = len(ok)
    correct = sum(r["correct"] for r in ok)
    print("=" * 64)
    print(f"classified={n}  correct={correct}  accuracy={correct / n * 100:.2f}%")
    gt = Counter(r["ground_truth"] for r in ok)
    for cls in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        members = [r for r in ok if r["ground_truth"] == cls]
        if not members:
            continue
        recall = sum(r["correct"] for r in members) / len(members) * 100
        print(f"  recall {cls:9}: {sum(r['correct'] for r in members)}/{len(members)} = {recall:.1f}%")
    # precision per predicted class
    pred = Counter(r["predicted"] for r in ok)
    for cls in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        members = [r for r in ok if r["predicted"] == cls]
        if not members:
            continue
        prec = sum(r["predicted"] == r["ground_truth"] for r in members) / len(members) * 100
        print(f"  precision {cls:6}: {sum(r['predicted'] == r['ground_truth'] for r in members)}/{len(members)} = {prec:.1f}%")
    print("confusion (rows=ground truth, cols=predicted):")
    classes = ("POSITIVE", "NEUTRAL", "NEGATIVE")
    print("GT\\PRED  " + "  ".join(f"{c:9}" for c in classes))
    for g in classes:
        row = [sum(1 for r in ok if r["ground_truth"] == g and r["predicted"] == p) for p in classes]
        print(f"{g:9} " + "  ".join(f"{v:9}" for v in row))
    errs = [r for r in ok if not r["correct"]]
    for r in errs[:20]:
        print(f"  err [{r['file_index']}] star={r['rating']} gt={r['ground_truth']} pred={r['predicted']} | {r['title'][:45]!r}")
    print("=" * 64)


if __name__ == "__main__":
    main()
