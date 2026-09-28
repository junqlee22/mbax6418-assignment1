"""Step 2b: re-score the first 100 reviews with the 3-class prompt (like-for-like).

The Step-2 run (run_eval.py) uses the binary POSITIVE/NEGATIVE prompt, exactly
as the assignment prescribes for that step. Step 6 then redefines the task to
three classes (4-5 -> POSITIVE, 3 -> NEUTRAL, 1-2 -> NEGATIVE), and the report
compares the first-100 slice against the balanced sample. Comparing the binary
first-100 score (97.0%) straight to the balanced 3-class score (77.3%) mixes
two changes: the added NEUTRAL class and the equal-weight sampling. This script
re-scores the SAME first 100 reviews (file order) with the SAME full 3-class
prompt used by the balanced run, so the two effects separate cleanly:

    first-100 binary 97.0%  ->  first-100 3-class X%  ->  balanced 3-class 77.3%

The star rating is NEVER shown to the model; ground truth is derived from the
rating after classification (same mapping as run_balanced.py).

Outputs:
    results_100_3class.csv                 parsed results (sentiment only used)
    outputs/first100_3class_raw.jsonl      raw model output per review

Resumable: file indexes already in the raw output file are skipped on re-run.
"""

import csv
import gzip
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from review_classifier import classify_full

SRC = "Gift_Cards.jsonl.gz"
CSV_OUT = "results_100_3class.csv"
RAW_OUT = "outputs/first100_3class_raw.jsonl"
N = 100
WORKERS = 4
RETRIES = 3


def load_rows(path: str, n: int) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows


def three_class(rating: float) -> str:
    if rating >= 4:
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def classify_one(review: dict) -> dict:
    last_err = None
    for attempt in range(RETRIES):
        try:
            parsed, raw = classify_full(
                review["title"], review["text"], timeout=90.0, return_raw=True
            )
            return {
                "predicted": parsed["sentiment"],
                "emotion_llm": parsed["emotion"],
                "raw": raw,
                "error": "",
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < RETRIES - 1:
                time.sleep(1.5 * (attempt + 1))
    return {"predicted": "", "emotion_llm": "", "raw": "", "error": repr(last_err)}


def main() -> None:
    rows = load_rows(SRC, N)
    os.makedirs("outputs", exist_ok=True)

    # Resume: indexes already stored in the raw file are not re-requested.
    done: dict[int, dict] = {}
    if os.path.exists(RAW_OUT):
        with open(RAW_OUT, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done[int(obj["index"])] = obj
    todo = [i for i in range(N) if i not in done]
    print(f"already done: {N - len(todo)}  to classify: {len(todo)}")

    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(classify_one, rows[i]): i for i in todo}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    rec = fut.result()
                    with open(RAW_OUT, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps({"index": i, **rec}, ensure_ascii=False) + "\n"
                        )
                except Exception as e:  # noqa: BLE001
                    print("worker failure:", e)

    # Rebuild the CSV from the raw file so a resume never loses completed rows.
    recs: dict[int, dict] = {}
    if os.path.exists(RAW_OUT):
        with open(RAW_OUT, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                recs[int(obj["index"])] = obj

    results = []
    for i, r in enumerate(rows):
        rating = float(r.get("rating"))
        gt = three_class(rating)
        rec = recs.get(i, {})
        pred = rec.get("predicted", "")
        results.append(
            {
                "index": i,
                "title": (r.get("title") or "").strip(),
                "text": (r.get("text") or "").strip(),
                "rating": rating,
                "ground_truth": gt,
                "predicted": pred,
                "emotion_llm": rec.get("emotion_llm", ""),
                "error": rec.get("error", ""),
                "correct": pred == gt,
            }
        )

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "index", "title", "text", "rating", "ground_truth",
                "predicted", "emotion_llm", "error", "correct",
            ],
        )
        w.writeheader()
        w.writerows(results)
    print(f"saved -> {CSV_OUT}")

    print("=" * 60)
    ok = [r for r in results if r["predicted"]]
    if not ok:
        print("no successfully classified rows -- check API access / errors")
        return
    c = sum(r["correct"] for r in ok)
    print(f"first-100 (3-class): classified={len(ok)} correct={c} accuracy={c/len(ok)*100:.2f}%")
    print("GT distribution:", dict(Counter(r["ground_truth"] for r in ok)))
    for cls in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        mem = [r for r in ok if r["ground_truth"] == cls]
        if mem:
            print(f"  recall {cls:9}: {sum(r['correct'] for r in mem)}/{len(mem)}")
    errs = [r for r in ok if not r["correct"]]
    for r in errs:
        print(f"  err idx={r['index']} star={r['rating']} gt={r['ground_truth']} "
              f"pred={r['predicted']} | {r['title'][:40]!r}")
    failed = [r for r in results if r["error"]]
    if failed:
        print("errored rows:", len(failed))
    print("=" * 60)


if __name__ == "__main__":
    main()
