"""Reusable review sentiment + emotion classifier via an OpenAI-compatible LLM.

Two modes:

- ``classify_review(title, text)`` -- Step 1/2 binary mode. Returns
  'POSITIVE' or 'NEGATIVE'. The prompt and parsing are kept byte-for-byte
  stable so the first-100-row run stays reproducible.

- ``classify_full(title, text)`` -- Steps 5/6 full mode. Returns
  ``{"sentiment": ..., "emotion": ...}`` where sentiment is one of
  POSITIVE/NEUTRAL/NEGATIVE and emotion is one of the eight NRC emotion
  names. The model emits strict JSON so both values are machine-readable.

Neither mode ever sees the star rating.
"""

import json
import re

import requests

_LLM_URL = "http://dobolyi.com:9001/v1/chat/completions"
_LLM_API_KEY = "6418"
_LLM_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

SENTIMENTS = ("POSITIVE", "NEUTRAL", "NEGATIVE")
EMOTIONS = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)

# --- Prompts (also documented in prompt.md) ---------------------------------

_BINARY_SYSTEM_PROMPT = (
    "You are a strict review sentiment classifier. A review consists of a "
    "title and a text body. Classify the OVERALL sentiment expressed in the "
    "text as either POSITIVE or NEGATIVE only. "
    "Rules:\n"
    "- Ignore any star rating entirely; base your decision only on the words.\n"
    "- If the title and body disagree, trust the body/text (it carries more content).\n"
    "- If the review is very short, use whatever signal exists; do not assume a default.\n"
    "- Sarcasm, anger, or profanity alone do not decide the class: judge the "
    "underlying attitude toward the product.\n"
    "- Reply with EXACTLY one word: POSITIVE or NEGATIVE. No punctuation, no "
    "explanation, no extra text."
)

_FULL_SYSTEM_PROMPT = (
    "You are a strict review sentiment and emotion classifier. A review "
    "consists of a TITLE and a BODY (text). "
    "Classify the sentiment of the review as exactly one of: "
    "POSITIVE, NEUTRAL, NEGATIVE. "
    "Then pick its single primary emotion from exactly one of: "
    "anger, anticipation, disgust, fear, joy, sadness, surprise, trust. "
    "Rules:\n"
    "- You are never shown a star rating: the rating does not exist. Base "
    "everything only on the words.\n"
    "- If the title and body conflict, trust the body (it carries more content).\n"
    "- POSITIVE = clearly favorable or enthusiastic about the product. "
    "NEGATIVE = clearly unfavorable or dissatisfied. "
    "NEUTRAL = balanced, factual, lukewarm, or mixed with no strong lean "
    "(including mild praise with a real caveat, or a mild complaint with real praise).\n"
    "- Short, terse, angry, or sarcastic reviews: judge the underlying "
    "attitude toward the product; do not assume a default.\n"
    "- Pick exactly ONE primary emotion -- the strongest one expressed. "
    "If none fits, choose the closest fit.\n"
    '- Reply with ONLY a JSON object, no other text, in exactly this shape:\n'
    '{"sentiment": "POSITIVE|NEUTRAL|NEGATIVE", '
    '"emotion": "anger|anticipation|disgust|fear|joy|sadness|surprise|trust"}'
)

# Fallback emotion synonyms for the rare case the model returns a near-miss.
_EMOTION_SYNONYMS = {
    "happy": "joy",
    "happiness": "joy",
    "pleased": "joy",
    "delight": "joy",
    "content": "trust",
    "contentment": "trust",
    "grateful": "trust",
    "gratitude": "trust",
    "satisfied": "trust",
    "angry": "anger",
    "frustrated": "anger",
    "frustration": "anger",
    "annoyed": "anger",
    "annoyance": "anger",
    "irritated": "anger",
    "irritable": "anger",
    "scared": "fear",
    "worried": "fear",
    "worry": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "sad": "sadness",
    "disappointed": "sadness",
    "disappointment": "sadness",
    "dislike": "disgust",
    "gross": "disgust",
    "excited": "anticipation",
    "excitement": "anticipation",
    "hope": "anticipation",
    "surprised": "surprise",
    "amazed": "surprise",
    "love": "joy",
    "loved": "joy",
    "neutral": "trust",
}


def _build_user_message(title: str, text: str) -> str:
    """Never send an empty prompt; keep the rating out of the message."""
    title = (title or "").strip()
    text = (text or "").strip()
    if text and title:
        return f"TITLE: {title}\nBODY: {text}"
    if text:
        return f"BODY: {text}"
    return "BODY: (empty)"


def _call_llm(
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    timeout: float = 60.0,
) -> str:
    payload = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {_LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(_LLM_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# --- Binary mode (Steps 1-2) --------------------------------------------------


def classify_review(title: str, text: str, timeout: float = 60.0) -> str:
    """Classify a review as 'POSITIVE' or 'NEGATIVE' (unchanged Step-1 prompt)."""
    raw = _call_llm(_BINARY_SYSTEM_PROMPT, _build_user_message(title, text), 2048, timeout)
    return _normalize_binary(raw)


def _normalize_binary(raw: str) -> str:
    m = re.search(r"\b(POSITIVE|NEGATIVE)\b", raw)
    if m:
        return m.group(1)
    if "POS" in raw and "NEG" not in raw:
        return "POSITIVE"
    if "NEG" in raw and "POS" not in raw:
        return "NEGATIVE"
    raise ValueError(f"Unparseable model output: {raw!r}")


# --- Full mode (Steps 5-6): sentiment + primary emotion ------------------------


def classify_full(title: str, text: str, timeout: float = 60.0) -> dict:
    """Classify sentiment (3 classes) and primary emotion. Returns a dict.

    Raises ValueError if the model output cannot be parsed into valid values.
    """
    raw = _call_llm(_FULL_SYSTEM_PROMPT, _build_user_message(title, text), 4096, timeout)
    return _parse_full(raw)


def _parse_full(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in model output: {raw[:200]!r}")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model output: {raw[:200]!r}") from e

    sentiment = str(obj.get("sentiment", "")).strip().upper()
    sentiment = _normalize_sentiment(sentiment)
    emotion = _normalize_emotion(str(obj.get("emotion", "")).strip().lower())
    return {"sentiment": sentiment, "emotion": emotion}


def _normalize_sentiment(s: str) -> str:
    if s in SENTIMENTS:
        return s
    for label in SENTIMENTS:
        if label in s:
            return label
    raise ValueError(f"Unrecognized sentiment: {s!r}")


def _normalize_emotion(e: str) -> str:
    if e in EMOTIONS:
        return e
    if e in _EMOTION_SYNONYMS:
        return _EMOTION_SYNONYMS[e]
    for name in EMOTIONS:
        if name in e:
            return name
    raise ValueError(f"Unrecognized emotion: {e!r}")


if __name__ == "__main__":
    samples = [
        ("Five stars, love it", "This blender is amazing, the best I've ever owned!"),
        ("Terrible", "It broke after one use and support never answered. Do not buy."),
        ("4 stars but...", "works okay but shipping was painfully slow"),
        ("worthless", "Honestly the best purchase all year, genuinely excellent"),
        ("Angry!!!", "THIS THING SUUUUCKS, worst garbage ever!!!"),
        ("Okay I guess", "It does the job. Nothing special, nothing bad."),
        ("Easy to use", "Very easy to use. I wish I knew about it earlier"),
    ]
    for t, b in samples:
        try:
            r = classify_full(t, b)
            print(f"{r['sentiment']:9} {r['emotion']:13} | {t!r} | {b[:40]}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR        | {t!r} | {b[:40]} -> {e}")
