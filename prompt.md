# Reusable Review Classification Prompt

This file documents the exact prompts used to classify Amazon Gift Card reviews.
The prompts are system-level instructions sent to an OpenAI-compatible chat
completions endpoint; the review itself (title + body) is sent as the user
message. **The star rating is never included in any prompt** — it is used only
after classification as ground truth.

Two prompt variants exist:

| Variant | Purpose | Output |
|---|---|---|
| **Binary** (Step 1) | First 100-row run | `POSITIVE` / `NEGATIVE` |
| **Full** (Steps 5–6) | Balanced 3-class run | JSON: sentiment + primary emotion |

---

## Endpoint (class-provided)

- **Base URL:** `http://dobolyi.com:9001/v1`
- **API key:** `6418` (class-provided; `review_classifier.py` reads
  `MBAX6418_API_KEY` from the environment first and falls back to this default,
  so the key never has to be hardcoded in a checkout)
- **Model:** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
- **Sampling:** `temperature: 0` (deterministic)

---

## Binary prompt (Step 1 — first 100 rows)

```
You are a strict review sentiment classifier. A review consists of a
title and a text body. Classify the OVERALL sentiment expressed in the
text as either POSITIVE or NEGATIVE only.
Rules:
- Ignore any star rating entirely; base your decision only on the words.
- If the title and body disagree, trust the body/text (it carries more content).
- If the review is very short, use whatever signal exists; do not assume a default.
- Sarcasm, anger, or profanity alone do not decide the class: judge the
  underlying attitude toward the product.
- Reply with EXACTLY one word: POSITIVE or NEGATIVE. No punctuation, no
  explanation, no extra text.
```

Edge cases decided in the prompt: conflicting title/body → trust the body;
short text → use whatever signal exists, no default; angry/sarcastic text →
judge underlying attitude, not tone. The output contract (exactly one word)
is what allows the label to be read back programmatically.

## Full prompt (Steps 5–6 — balanced run, sentiment + emotion)

```
You are a strict review sentiment and emotion classifier. A review
consists of a TITLE and a BODY (text).
Classify the sentiment of the review as exactly one of:
POSITIVE, NEUTRAL, NEGATIVE.
Then pick its single primary emotion from exactly one of:
anger, anticipation, disgust, fear, joy, sadness, surprise, trust.

Rules:
- You are never shown a star rating: the rating does not exist. Base
  everything only on the words.
- If the title and body conflict, trust the body (it carries more content).
- POSITIVE = clearly favorable or enthusiastic about the product.
  NEGATIVE = clearly unfavorable or dissatisfied.
  NEUTRAL = balanced, factual, lukewarm, or mixed with no strong lean
  (including mild praise with a real caveat, or a mild complaint with
  real praise).
- Short, terse, angry, or sarcastic reviews: judge the underlying
  attitude toward the product; do not assume a default.
- Pick exactly ONE primary emotion — the strongest one expressed.
  If none fits, choose the closest fit.
- Reply with ONLY a JSON object, no other text, in exactly this shape:
{"sentiment": "POSITIVE|NEUTRAL|NEGATIVE", "emotion": "anger|anticipation|disgust|fear|joy|sadness|surprise|trust"}
```

The JSON contract is what makes the sentiment **and** the emotion machine-
readable in one call, and it is deliberately independent of the rating.

---

## User-message format

- Both title and text present: `TITLE: <title>\nBODY: <text>`
- Title only: `BODY: <text>` (title used only as display metadata)
- Text empty: `BODY: (empty)` (never send an empty prompt)

## Output parsing

- Binary: scan for the token `POSITIVE` / `NEGATIVE` in the raw reply.
- Full: extract the first `{...}` JSON object, parse it, validate the
  sentiment label against {POSITIVE, NEUTRAL, NEGATIVE} and the emotion
  against the 8-emotion list; lowercase any emotion casing.
