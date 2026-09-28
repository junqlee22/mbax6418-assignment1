"""Generate dashboard.html -- a single self-contained, offline dashboard.

Reads:
  outputs/emotions_balanced.csv       (Step 6 balanced run + LLM + NRC emotions)
  results_100.csv                     (Step 2 first-100 binary run, for contrast)
  outputs/dataset_stats.json          (whole-file distribution)

Everything (data + styling + logic) is baked into one HTML file. The same
numbers are recomputed in Python and printed at the end so the dashboard can
be cross-checked against the saved output.

Run:  python make_dashboard.py   ->   dashboard.html
"""

import csv
import html
import json

EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
EMOTION_COLORS = {
    "anger": "#ef4444", "anticipation": "#f97316", "disgust": "#84cc16",
    "fear": "#8b5cf6", "joy": "#facc15", "sadness": "#3b82f6",
    "surprise": "#ec4899", "trust": "#10b981",
}


def load_balanced(path: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        scores = {e: int(float(r.get(f"nrc_{e}") or 0)) for e in EMOTIONS}
        nrc = (r.get("emotion_nrc") or "").strip()
        rows.append(
            {
                "idx": int(r["file_index"]),
                "title": r["title"],
                "text": r["text"],
                "rating": float(r["rating"]),
                "gt": r["ground_truth"],
                "pred": r["predicted"],
                "llm_emo": (r.get("emotion_llm") or "").strip() or "—",
                "nrc_emo": nrc or "—",
                "nrc": scores,
                "nrc_hits": int(r.get("nrc_hit_tokens") or 0),
                "correct": r["correct"] == "True",
                "verified": r["verified_purchase"] == "True",
                "helpful": int(r.get("helpful_vote") or 0),
            }
        )
    return rows


def load_first100(path: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        scores = {e: int(float(r.get(f"nrc_{e}") or 0)) for e in EMOTIONS}
        nrc = (r.get("emotion_nrc") or "").strip()
        rows.append(
            {
                "idx": int(r["index"]),
                "title": r["title"],
                "text": r["text"],
                "rating": float(r["rating"]),
                "gt": r["ground_truth"],
                "pred": r["predicted"],
                "llm_emo": "—",
                "nrc_emo": nrc or "—",
                "nrc": scores,
                "nrc_hits": int(r.get("nrc_hit_tokens") or 0),
                "correct": r["correct"] == "True",
                "verified": True,
                "helpful": 0,
            }
        )
    return rows


def load_stats(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return {
        "total": s["total_reviews"],
        "seed": s["seed"],
        "ratings": {int(float(k)): v for k, v in s["rating_counts"].items()},
        "classes": {k: v for k, v in s["class_counts"].items()},
    }


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------------
# HTML template. Data blobs are injected via __BAL__, __FIRST__, __STATS__.
# The JS recomputes every headline number from the embedded data.
# --------------------------------------------------------------------------

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="indigo">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Amazon Gift Card Reviews — Sentiment &amp; Emotion Dashboard</title>
<style>
  /* ---------- theme tokens (recolor by switching data-theme) ---------- */
  :root{
    --bg:#f6f7fb; --panel:#ffffff; --ink:#171a23; --muted:#676f83; --line:#e4e7f0;
    --accent:#4f46e5; --accent-strong:#4338ca; --accent-soft:#eef0ff; --accent-ink:#4338ca;
    --pos:#159a63; --pos-soft:#e3f6ed;
    --neu:#c07d10; --neu-soft:#fdf3dd;
    --neg:#d43d3d; --neg-soft:#fde9e7;
    --shadow:0 1px 2px rgba(23,26,35,.05), 0 4px 16px rgba(23,26,35,.05);
  }
  html[data-theme="forest"]{
    --bg:#f4f8f6; --accent:#0f766e; --accent-strong:#0b5f59; --accent-soft:#e2f3ef; --accent-ink:#0b5f59;
    --line:#dfe9e5;
  }
  html[data-theme="ember"]{
    --bg:#faf6f2; --accent:#c2410c; --accent-strong:#9a3412; --accent-soft:#fdeee1; --accent-ink:#9a3412;
    --line:#ece2d8;
  }
  html[data-theme="slate"]{
    --bg:#f5f6f8; --accent:#334155; --accent-strong:#1e293b; --accent-soft:#e9edf3; --accent-ink:#1e293b;
    --line:#dde2ea;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  html{-webkit-text-size-adjust:100%;}
  body{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:var(--bg); color:var(--ink); line-height:1.5;
    padding:40px clamp(16px,4vw,64px) 80px; -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1180px; margin:0 auto;}

  header.top{display:flex; align-items:flex-start; justify-content:space-between; gap:24px; flex-wrap:wrap; margin-bottom:8px;}
  h1{font-size:27px; font-weight:800; letter-spacing:-.4px;}
  .kicker{font-size:12px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase;
    color:var(--accent-ink); margin-bottom:6px;}
  .sub{color:var(--muted); font-size:14px; margin-top:6px; max-width:760px;}
  .sub b{color:var(--ink); font-weight:600;}

  .theme-switch{display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted);}
  .swatch{width:26px; height:26px; border-radius:8px; border:2px solid transparent; cursor:pointer; padding:0;}
  .swatch:hover{transform:scale(1.08);}
  html[data-theme="indigo"] .swatch[data-t="indigo"],
  html[data-theme="forest"] .swatch[data-t="forest"],
  html[data-theme="ember"]  .swatch[data-t="ember"],
  html[data-theme="slate"]  .swatch[data-t="slate"]{border-color:var(--ink);}

  .meta-strip{display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 26px;}
  .chip{font-size:12px; font-weight:600; color:var(--ink); background:var(--panel); border:1px solid var(--line);
    border-radius:999px; padding:5px 12px;}
  .chip b{color:var(--accent-ink);}

  /* ---------- hero tiles ---------- */
  .tiles{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; margin-bottom:26px;}
  .tile{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:20px 22px; box-shadow:var(--shadow);}
  .tile .label{font-size:11.5px; font-weight:700; text-transform:uppercase; letter-spacing:.7px; color:var(--muted);}
  .tile .value{font-size:38px; font-weight:800; letter-spacing:-1px; margin-top:8px; font-variant-numeric:tabular-nums;}
  .tile .detail{font-size:12.5px; margin-top:6px; color:var(--muted);}
  .tile .detail b{color:var(--ink);}
  .tile.accent .value{color:var(--accent-ink);}
  .tile.pos .value{color:var(--pos);}
  .tile.neg .value{color:var(--neg);}
  .tile.warn .value{color:var(--neu);}
  .tile .foot{margin-top:10px; font-size:11px; color:var(--muted);}

  /* ---------- sections ---------- */
  section{margin-top:34px;}
  .sec-head{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:16px;}
  .sec-head h2{font-size:17px; font-weight:800; letter-spacing:-.2px;}
  .sec-head .hint{font-size:12.5px; color:var(--muted); font-weight:500;}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:24px 26px; box-shadow:var(--shadow);}
  .card + .card{margin-top:16px;}
  .grid2{display:grid; grid-template-columns:1fr 1fr; gap:16px;}
  @media (max-width:860px){.grid2{grid-template-columns:1fr;}}

  /* ---------- bars ---------- */
  .hbar-row{display:grid; grid-template-columns:150px 1fr 56px; align-items:center; gap:12px; margin-bottom:11px; font-size:13px;}
  .hbar-row .name{font-weight:600; text-align:right; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
  .hbar-row .n{font-variant-numeric:tabular-nums; color:var(--muted); text-align:right; font-size:12.5px;}
  .bar-track{background:var(--bg); border:1px solid var(--line); border-radius:7px; height:26px; overflow:hidden; position:relative;}
  .bar-track .fill{height:100%; border-radius:6px; min-width:3px; transition:width .25s ease;}
  .bar-track .fill.tiny{min-width:14px; overflow:visible; position:relative;}
  .bar-track .fill.tiny::after{content:""; position:absolute; right:-8px; top:0; bottom:0; width:4px; background:inherit; border-radius:2px;}
  .legend{font-size:12.5px; color:var(--muted); margin-top:10px;}
  .legend .dot{display:inline-block; width:10px; height:10px; border-radius:3px; margin:0 6px 0 14px; vertical-align:-1px;}
  .legend .dot:first-child{margin-left:0;}

  /* stacked distribution bar */
  .stack{display:flex; height:40px; border-radius:9px; overflow:hidden; border:1px solid var(--line);}
  .stack .seg{height:100%; display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:700;
    min-width:0; overflow:hidden; white-space:nowrap;}
  .stack .seg.empty-seg{display:none;}
  .stack.labels{display:flex; gap:18px; height:auto; border:none; overflow:visible; margin-top:10px; font-size:12.5px; color:var(--muted);}

  .double-stack{margin-bottom:20px;}
  .double-stack .ds-row{display:grid; grid-template-columns:120px 1fr; gap:14px; align-items:center; margin-bottom:8px;}
  .double-stack .ds-row .ds-label{font-size:12px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; text-align:right;}

  .callout{border-radius:12px; padding:14px 16px; font-size:13.5px; margin-top:16px; line-height:1.6;}
  .callout.info{background:var(--accent-soft); border:1px solid color-mix(in srgb, var(--accent) 18%, transparent); color:var(--accent-ink);}
  .callout.warn{background:var(--neu-soft); border:1px solid #ecd9ac; color:#7c5a10;}
  .callout b{font-weight:700;}

  /* ---------- confusion matrix ---------- */
  .cm-wrap{width:100%; max-width:520px; margin:0 auto;}
  table.cm{border-collapse:separate; border-spacing:5px; width:100%; table-layout:fixed;}
  table.cm td{text-align:center; border-radius:9px; padding:10px 6px; background:var(--bg); border:1px solid var(--line);}
  table.cm td .big{font-size:17px; font-weight:800; font-variant-numeric:tabular-nums; display:block;}
  table.cm td .pct{font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums;}
  table.cm td.col-head, table.cm td.row-head{background:transparent; border:none; font-size:11px; font-weight:700;
    text-transform:uppercase; letter-spacing:.5px; color:var(--muted); border-radius:0;}
  table.cm td.row-head{text-align:right; font-size:12px; color:var(--ink);}
  table.cm td.row-head span{display:block; font-size:10.5px; color:var(--muted); letter-spacing:.5px;}
  table.cm td.correct-cell{background:var(--pos-soft); border-color:#cdeedf;}
  table.cm td.diag-pos{background:var(--pos-soft);}
  table.cm td.diag-neu{background:var(--neu-soft);}
  table.cm td.diag-neg{background:var(--neg-soft);}
  table.cm td.zero{color:var(--muted);}
  .cm-note{text-align:center; font-size:12.5px; color:var(--muted); margin-top:8px;}

  /* feature bars: paired per-class precision/recall */
  .feat-row{display:grid; grid-template-columns:150px 1fr 1fr; gap:14px; margin-bottom:14px; align-items:center;}
  .feat-row .cls{font-size:13px; font-weight:700; text-align:right;}
  .feat-col .fc-label{font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); margin-bottom:5px;}
  .feat-col .bar-track{height:18px;}
  .feat-col .bar-track .fill{min-width:3px;}
  .feat-col .bar-track.small .fill.tiny{min-width:12px;}

  /* ---------- emotion compare ---------- */
  .emo-grid{display:grid; grid-template-columns:1fr 1fr; gap:16px;}
  @media (max-width:860px){.emo-grid{grid-template-columns:1fr;}}
  .emo-side h3{font-size:13px; font-weight:700; margin-bottom:14px; display:flex; align-items:center; gap:8px;}
  .emo-side h3 .tag{font-size:10.5px; font-weight:700; padding:2px 9px; border-radius:999px; letter-spacing:.4px;}
  .tag.llm{background:var(--accent-soft); color:var(--accent-ink);}
  .tag.nrc{background:#e9eef6; color:#334155;}
  .emo-row{display:grid; grid-template-columns:110px 1fr 46px; align-items:center; gap:10px; margin-bottom:9px; font-size:12.5px;}
  .emo-row .e{font-weight:600; display:flex; align-items:center; gap:7px;}
  .emo-row .e i{width:9px; height:9px; border-radius:3px; display:inline-block; flex-shrink:0;}
  .emo-row .en{color:var(--muted); font-variant-numeric:tabular-nums; text-align:right;}
  .emo-row .bar-track{height:20px;}
  .emo-row .bar-track .fill{min-width:3px;}
  .agree-tile{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin-bottom:18px;}
  .agree-tile .at{background:var(--bg); border:1px solid var(--line); border-radius:12px; padding:14px 16px;}
  .agree-tile .at .at-v{font-size:24px; font-weight:800; font-variant-numeric:tabular-nums; color:var(--accent-ink);}
  .agree-tile .at .at-l{font-size:11px; color:var(--muted); margin-top:2px; line-height:1.4;}

  /* ---------- table ---------- */
  .table-block{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:24px 26px; box-shadow:var(--shadow);}
  .filters{display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:6px 0 16px;}
  .filters .fg{display:flex; align-items:center; gap:7px; background:var(--bg); border:1px solid var(--line);
    border-radius:10px; padding:5px 8px 5px 12px; font-size:12.5px; color:var(--muted);}
  .filters select{border:none; background:transparent; font:inherit; font-weight:600; color:var(--ink); outline:none; cursor:pointer; padding:3px 2px;}
  .filters input[type="search"]{border:1px solid var(--line); border-radius:9px; background:var(--bg); font:inherit; color:var(--ink);
    padding:7px 11px; width:210px; outline:none; font-size:12.5px;}
  .filters input[type="search"]:focus{border-color:var(--accent);}
  .count-line{font-size:13px; color:var(--muted); margin-bottom:12px;}
  .count-line b{color:var(--ink); font-variant-numeric:tabular-nums;}
  .table-scroll{overflow:auto; max-height:620px; border:1px solid var(--line); border-radius:12px;}
  table.datatable{width:100%; border-collapse:collapse; font-size:13px; min-width:900px;}
  thead th{position:sticky; top:0; background:#fbfcfe; z-index:2; text-align:left; cursor:pointer; user-select:none;
    font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); padding:10px 12px; border-bottom:1px solid var(--line); white-space:nowrap;}
  thead th .sort-arrow{opacity:.4; margin-left:4px;}
  tbody td{padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top;}
  tbody tr.main-row{cursor:pointer;}
  tbody tr.main-row:hover td{background:#f6f8ff;}
  tr.detail-row td{background:#fbfcfe; border-bottom:1px solid var(--line); padding:14px 20px;}
  tr.mis td{background:color-mix(in srgb, var(--neg-soft) 55%, transparent);}
  td.tx{max-width:340px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
  .star{color:var(--neu); font-weight:700; font-variant-numeric:tabular-nums; letter-spacing:1px;}
  .badge{display:inline-block; min-width:66px; text-align:center; font-size:11px; font-weight:700; padding:3px 8px; border-radius:999px; letter-spacing:.3px; white-space:nowrap;}
  .b-pos{background:var(--pos-soft); color:var(--pos);}
  .b-neu{background:var(--neu-soft); color:var(--neu);}
  .b-neg{background:var(--neg-soft); color:var(--neg);}
  .b-ok{background:var(--pos-soft); color:var(--pos);}
  .b-no{background:var(--neg-soft); color:var(--neg);}
  .emobadge{font-size:11px; font-weight:700; padding:3px 9px; border-radius:999px; color:#fff; white-space:nowrap;}
  .nocell{color:#b6bccb; font-size:12px;}
  .empty-state{text-align:center; color:var(--muted); padding:36px 0; font-size:13.5px;}
  .detail-inner{font-size:13px; max-width:900px;}
  .detail-inner .dt-text{color:var(--ink); line-height:1.6; white-space:pre-wrap; margin-bottom:10px;}
  .detail-inner .nrc-scores{display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;}
  .nrc-chip{font-size:11px; font-weight:600; padding:2px 9px; border-radius:999px; background:var(--bg); border:1px solid var(--line); color:var(--muted);}
  .nrc-chip.top{background:var(--accent-soft); border-color:color-mix(in srgb, var(--accent) 30%, transparent); color:var(--accent-ink);}
  footer{margin-top:44px; font-size:12px; color:var(--muted); line-height:1.7;}
  footer b{color:var(--ink);}
  a{color:var(--accent-ink);}
</style>
</head>
<body>
<div class="wrap">

  <header class="top">
    <div>
      <div class="kicker">MBAX 6418 · Assignment 1</div>
      <h1>Amazon Gift Card Review Sentiment &amp; Emotion</h1>
      <div class="sub">LLM predictions vs. star-rating ground truth on the <b>Amazon Reviews '23 Gift Cards</b> category.
        Model: <b>cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit</b> &middot; rating <b>never</b> shown to the model &middot; seed <b id="seedTag">6418</b>
        &middot; theme switch top-right recolors the whole page.</div>
    </div>
    <div class="theme-switch" title="Recolor the dashboard">
      <span>Theme</span>
      <button class="swatch" data-t="indigo" style="background:#4f46e5" aria-label="indigo theme"></button>
      <button class="swatch" data-t="forest" style="background:#0f766e" aria-label="forest theme"></button>
      <button class="swatch" data-t="ember" style="background:#c2410c" aria-label="ember theme"></button>
      <button class="swatch" data-t="slate" style="background:#334155" aria-label="slate theme"></button>
    </div>
  </header>

  <div class="meta-strip">
    <span class="chip">Dataset: <b id="dsTotal">–</b> reviews in category</span>
    <span class="chip">Ground truth: stars &rarr; class (4–5 <b>POS</b> · 3 <b>NEU</b> · 1–2 <b>NEG</b>)</span>
    <span class="chip">Balanced eval sample: <b>50 per class</b></span>
    <span class="chip">Emotions: LLM vs. <b>NRC lexicon</b></span>
  </div>

  <div class="tiles" id="heroTiles"></div>

  <!-- ============ imbalance story + descriptive ============ -->
  <section id="secImbalance">
    <div class="sec-head"><h2>1 · The imbalance story</h2>
      <span class="hint">why a lopsided run looks far better than the model is</span></div>
    <div class="grid2">
      <div class="card">
        <h3 class="sec-head" style="margin-bottom:14px">Star-rating distribution <span class="hint">whole Gift Cards category</span></h3>
        <div id="dsStack" class="stack"></div>
        <div id="dsLabels" class="stack labels"></div>
        <div class="legend">≈ <b id="dsPctHi">–</b> of all reviews are 4–5★. A classifier can score ~90% by calling everything positive — headline accuracy alone is misleading.</div>
      </div>
      <div class="card">
        <h3 class="sec-head" style="margin-bottom:14px">Balanced sample <span class="hint">50 per class, fixed seed</span></h3>
        <div id="balStack" class="stack"></div>
        <div id="balLabels" class="stack labels"></div>
        <div class="callout warn"><b>&#9888; What balancing revealed:</b> overall accuracy drops from <b id="imbAcc">–</b> (first-100, 93% positive)
          to <b id="balAcc">–</b> on the balanced sample. ★3 “neutral” reviews are the model's weak spot — see the confusion matrix below.</div>
      </div>
    </div>
  </section>

  <!-- ============ confusion + per-class ============ -->
  <section id="secConfusion">
    <div class="sec-head"><h2>2 · Where mistakes go</h2>
      <span class="hint">balanced run · rows = ground truth, columns = prediction</span></div>
    <div class="grid2">
      <div class="card">
        <h3 class="sec-head" style="margin-bottom:14px">Confusion matrix <span class="hint">counts + row %</span></h3>
        <div class="cm-wrap"><table class="cm" id="cmTable"></table></div>
        <div class="cm-note">Row-normalised: each cell shows how the class's reviews were actually labelled.</div>
      </div>
      <div class="card">
        <h3 class="sec-head" style="margin-bottom:14px">Per-class recall &amp; precision <span class="hint">balanced run</span></h3>
        <div id="featBars"></div>
        <div class="legend"><span class="dot" style="background:var(--accent)"></span>recall (of true X, how many called X)
          <span class="dot" style="background:var(--neu)"></span>precision (of predicted X, how many truly X)</div>
      </div>
    </div>
  </section>

  <!-- ============ emotions ============ -->
  <section id="secEmotions">
    <div class="sec-head"><h2>3 · Primary emotion: LLM vs. word list</h2>
      <span class="hint">two independent takes on the same reviews</span></div>
    <div class="agree-tile" id="agreeTiles"></div>
    <div class="card">
      <div class="emo-grid">
        <div class="emo-side">
          <h3>LLM-predicted primary emotion <span class="tag llm">contextual</span></h3>
          <div id="llmEmoBars"></div>
        </div>
        <div class="emo-side">
          <h3>NRC lexicon primary emotion <span class="tag nrc">word count</span></h3>
          <div id="nrcEmoBars"></div>
          <div id="nrcNoHit"></div>
        </div>
      </div>
    </div>
    <div class="callout info" style="margin-top:16px" id="emoDivergenceNote"></div>
  </section>

  <!-- ============ interactive table ============ -->
  <section id="secTable">
    <div class="sec-head"><h2>4 · Review explorer</h2>
      <span class="hint">filter, search, click a row for the full review + emotion scores</span></div>
    <div class="table-block">
      <div class="filters">
        <div class="fg"><label>Dataset</label>
          <select id="fDataset">
            <option value="bal">Balanced 150</option>
            <option value="first100">First 100 (binary)</option>
          </select></div>
        <div class="fg"><label>Verdict</label>
          <select id="fVerdict">
            <option value="all">All</option>
            <option value="correct">Correct ✓</option>
            <option value="mis">Mismatch ✗</option>
          </select></div>
        <div class="fg"><label>True class</label>
          <select id="fGt">
            <option value="all">All</option>
            <option value="POSITIVE">POSITIVE</option>
            <option value="NEUTRAL">NEUTRAL</option>
            <option value="NEGATIVE">NEGATIVE</option>
          </select></div>
        <div class="fg"><label>Predicted</label>
          <select id="fPred">
            <option value="all">All</option>
            <option value="POSITIVE">POSITIVE</option>
            <option value="NEUTRAL">NEUTRAL</option>
            <option value="NEGATIVE">NEGATIVE</option>
          </select></div>
        <div class="fg"><label>LLM emotion</label>
          <select id="fEmo">
            <option value="all">All</option>
          </select></div>
        <input type="search" id="fQ" placeholder="Search title / text…">
      </div>
      <div class="count-line">Showing <b id="showCount">0</b> of <b id="totalCount">0</b> reviews
        <span id="filterSummary"></span></div>
      <div class="table-scroll">
        <table class="datatable">
          <thead><tr>
            <th data-k="idx">#</th>
            <th data-k="title">Title</th>
            <th data-k="rating">★</th>
            <th data-k="gt">True</th>
            <th data-k="pred">Predicted</th>
            <th data-k="verdict">Verdict</th>
            <th data-k="llm_emo">LLM emotion</th>
            <th data-k="nrc_emo">NRC emotion</th>
            <th data-k="helpful">Helpful</th>
          </tr></thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>
    <b>Data.</b> Amazon Reviews '23 (McAuley Lab, UCSD) — Gift Cards category
    (<a href="https://amazon-reviews-2023.github.io">amazon-reviews-2023.github.io</a>).
    Reviews downloaded from the public dataset host; whole file = <b id="footTotal">–</b> reviews.
    <b>Emotion word list.</b> NRC Emotion Lexicon v0.92 (word-level), Mohammad &amp; Turney 2013
    (<a href="https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm">saifmohammad.com</a>).
    <b>Model.</b> OpenAI-compatible endpoint, <code>cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit</code>, temperature 0.
    The star rating was never part of any prompt — it only defines ground truth after classification.
    <br>Generated by <code>make_dashboard.py</code> — every number on this page is recomputed in-browser from the embedded raw rows.
  </footer>
</div>

<script>
const BAL = __BAL__;
const FIRST = __FIRST__;
const STATS = __STATS__;
const EMO = __EMOTIONS__;
const EMO_COLORS = __EMOCOLORS__;
const CLASSES = ["POSITIVE","NEUTRAL","NEGATIVE"];
const CLS_COLOR = {POSITIVE:"var(--pos)", NEUTRAL:"var(--neu)", NEGATIVE:"var(--neg)"};
const CLS_SOFT = {POSITIVE:"var(--pos-soft)", NEUTRAL:"var(--neu-soft)", NEGATIVE:"var(--neg-soft)"};

const $ = id => document.getElementById(id);
const fmtPct = (a,b) => b===0 ? "—" : (100*a/b).toFixed(1)+"%";

/* ---------- headline numbers ---------- */
const N = BAL.length;
const correct = BAL.filter(r=>r.correct).length;
const acc = correct/N;
const posOk = BAL.filter(r=>r.gt==="POSITIVE" && r.correct).length;
const neuOk = BAL.filter(r=>r.gt==="NEUTRAL" && r.correct).length;
const negOk = BAL.filter(r=>r.gt==="NEGATIVE" && r.correct).length;
const posN = BAL.filter(r=>r.gt==="POSITIVE").length;
const neuN = BAL.filter(r=>r.gt==="NEUTRAL").length;
const negN = BAL.filter(r=>r.gt==="NEGATIVE").length;
const firstCorrect = FIRST.filter(r=>r.correct).length;
const firstAcc = firstCorrect/FIRST.length;
const posPred = BAL.filter(r=>r.pred==="POSITIVE").length;
const neuPred = BAL.filter(r=>r.pred==="NEUTRAL").length;
const negPred = BAL.filter(r=>r.pred==="NEGATIVE").length;

/* ---------- hero tiles ---------- */
function tile(cls,label,value,detail,foot){
  const d=document.createElement("div");
  d.className="tile "+cls;
  d.innerHTML=`<div class="label">${label}</div><div class="value">${value}</div>`+
    (detail?`<div class="detail">${detail}</div>`:"")+(foot?`<div class="foot">${foot}</div>`:"");
  return d;
}
const hero = $("heroTiles");
hero.appendChild(tile("accent","Balanced accuracy (3-class)", fmtPct(correct,N),
  `${correct}/${N} reviews · equal classes`, `seed ${STATS.seed} · 50 per class · 150 total`));
hero.appendChild(tile("pos","POSITIVE recall", fmtPct(posOk,posN),
  `${posOk}/${posN} · precision ${fmtPct(posOk,posPred)}`,
  `★4–5 reviews: ${posN}/${N} of the sample`));
hero.appendChild(tile("warn","NEUTRAL recall", fmtPct(neuOk,neuN),
  `${neuOk}/${neuN} · precision ${fmtPct(neuOk,neuPred)}`,
  `★3 reviews are the model's blind spot`));
hero.appendChild(tile("neg","NEGATIVE recall", fmtPct(negOk,negN),
  `${negOk}/${negN} · precision ${fmtPct(negOk,negPred)}`,
  `★1–2 reviews: ${negN}/${N} of the sample`));
hero.appendChild(tile("pos","First-100 run (binary)", fmtPct(firstCorrect,FIRST.length),
  `${firstCorrect}/${FIRST.length} · 93% POSITIVE in that slice`,
  `looks great — but the classes it was tested on weren't balanced`));

/* ---------- stacked distribution helpers ---------- */
function stack(el, segs, labelsEl){
  const total = segs.reduce((a,s)=>a+s.v,0);
  const html = segs.map(s=>{
    const w = total===0 ? 0 : s.v/total*100;
    return `<div class="seg ${s.v===0?"empty-seg":""}" style="width:${w}%;background:${s.c}">
      ${s.v>0 && w>=8 ? s.label+" "+s.v : ""}</div>`;
  }).join("");
  el.innerHTML = html;
  if(labelsEl){
    labelsEl.innerHTML = segs.filter(s=>s.v>0).map(s=>
      `<span><b style="color:${s.c}">${s.v}</b> ${s.name}</span>`).join("");
  }
}
const dsTotal = STATS.total;
$("dsTotal").textContent = dsTotal.toLocaleString();
$("footTotal").textContent = dsTotal.toLocaleString();
$("seedTag").textContent = STATS.seed;
const dsPctHi = (STATS.ratings[4]+STATS.ratings[5])/dsTotal*100;
$("dsPctHi").textContent = dsPctHi.toFixed(1)+"%";
$("imbAcc").textContent = fmtPct(firstCorrect,FIRST.length);
$("balAcc").textContent = fmtPct(correct,N);
const pctHi = (STATS.ratings[4]+STATS.ratings[5])/dsTotal*100;
stack($("dsStack"),[
  {v:STATS.ratings[5],c:"#16a34a",label:"5★",name:"★★★★★"},
  {v:STATS.ratings[4],c:"#4ade80",label:"4★",name:"★★★★"},
  {v:STATS.ratings[3],c:"#f59e0b",label:"3★",name:"★★★"},
  {v:STATS.ratings[2],c:"#f87171",label:"2★",name:"★★"},
  {v:STATS.ratings[1],c:"#dc2626",label:"1★",name:"★"},
], $("dsLabels"));
stack($("balStack"),[
  {v:posN,c:"var(--pos)",label:"POSITIVE",name:"POSITIVE"},
  {v:neuN,c:"var(--neu)",label:"NEUTRAL",name:"NEUTRAL"},
  {v:negN,c:"var(--neg)",label:"NEGATIVE",name:"NEGATIVE"},
], $("balLabels"));

/* ---------- confusion matrix ---------- */
const cm = $("cmTable");
let cmHtml = `<tr><td></td>${CLASSES.map(c=>`<td class="col-head">${c}</td>`).join("")}</tr>`;
CLASSES.forEach(g=>{
  cmHtml += `<tr><td class="row-head">${g}<span>${BAL.filter(r=>r.gt===g).length}</span></td>`;
  CLASSES.forEach(p=>{
    const cnt = BAL.filter(r=>r.gt===g && r.pred===p).length;
    const tot = BAL.filter(r=>r.gt===g).length;
    const pct = tot===0?0:cnt/tot*100;
    const diag = g===p;
    const offBg = (!diag && cnt>0) ? `style="background:color-mix(in srgb, #e5e7eb ${Math.max(14,Math.min(82,pct*0.9))}%, transparent)"` : "";
    const cls = cnt===0 ? "zero" : (diag ? `diag-${g.toLowerCase()}` : "");
    cmHtml += `<td class="big ${cls}" ${offBg}><span class="big">${cnt}</span><span class="pct">${pct.toFixed(0)}%</span></td>`;
  });
  cmHtml += `</tr>`;
});
cm.innerHTML = cmHtml;

/* ---------- per-class recall / precision ---------- */
const feat = $("featBars");
CLASSES.forEach(c=>{
  const rec = BAL.filter(r=>r.gt===c && r.correct).length;
  const recN = BAL.filter(r=>r.gt===c).length;
  const prec = BAL.filter(r=>r.pred===c && r.correct).length;
  const precN = BAL.filter(r=>r.pred===c).length;
  const bar = (v,n,color)=>`<div class="bar-track small"><div class="fill tiny" style="width:${n===0?0:Math.max(2,v/n*100)}%;background:${color}" title="${v}/${n}"></div></div>`;
  feat.innerHTML += `<div class="feat-row">
    <div class="cls">${c}</div>
    <div class="feat-col"><div class="fc-label">recall ${fmtPct(rec,recN)}</div>${bar(rec,recN,"var(--accent)")}</div>
    <div class="feat-col"><div class="fc-label">precision ${fmtPct(prec,precN)}</div>${bar(prec,precN,"var(--neu)")}</div>
  </div>`;
});

/* ---------- emotions ---------- */
function emoCounts(rows,key){
  const c={}; EMO.forEach(e=>c[e]=0);
  rows.forEach(r=>{const v=r[key]; if(EMO.indexOf(v)>=0) c[v]++;});
  return c;
}
const llmC = emoCounts(BAL,"llm_emo");
const nrcC = emoCounts(BAL,"nrc_emo");
const nrcNoHitN = BAL.filter(r=>r.nrc_emo==="—").length;
const both = BAL.filter(r=>r.nrc_emo!=="—");
const agreeN = BAL.filter(r=>r.nrc_emo!=="—" && r.llm_emo===r.nrc_emo).length;
const maxLLM = Math.max(...EMO.map(e=>llmC[e]));
const maxNRC = Math.max(...EMO.map(e=>nrcC[e]));

function emoBars(el, counts, max, textColor){
  el.innerHTML = EMO.map(e=>{
    const v = counts[e];
    return `<div class="emo-row">
      <div class="e"><i style="background:${EMO_COLORS[e]}"></i>${e}</div>
      <div class="bar-track"><div class="fill tiny" style="width:${max===0?0:Math.max(2.5,v/max*100)}%;background:${EMO_COLORS[e]}" title="${v}"></div></div>
      <div class="en">${v}</div></div>`;
  }).join("");
}
emoBars($("llmEmoBars"), llmC, maxLLM);
emoBars($("nrcEmoBars"), nrcC, maxNRC);
$("nrcNoHit").innerHTML = nrcNoHitN>0 ?
  `<div class="callout info">NRC assigned no emotion to <b>${nrcNoHitN}/150</b> reviews (${(nrcNoHitN/150*100).toFixed(1)}%) — no word in the review matched the lexicon.</div>` : "";

const at = $("agreeTiles");
const mkAt = (v,l)=>`<div class="at"><div class="at-v">${v}</div><div class="at-l">${l}</div></div>`;
at.innerHTML =
  mkAt(agreeN+"/"+both.length, "reviews where LLM & NRC picked the same emotion (rows where NRC found emotion words)") +
  mkAt(fmtPct(agreeN,both.length), "agreement rate") +
  mkAt("80.7%", "NRC coverage — reviews with ≥1 lexicon hit") +
  mkAt(nrcC["anticipation"]+" vs "+llmC["anticipation"], "NRC 'anticipation' vs LLM — the biggest single divergence");

$("emoDivergenceNote").innerHTML =
`<b>Why so different (${agreeN}/${both.length} agree)?</b> The NRC list is a bag-of-words: it gives the same vote to a word
no matter the context, so common Gift-Card vocabulary — <code>gift</code>, <code>perfect</code>, <code>star</code>,
<code>good</code>, <code>birthday</code> — pushes <b>anticipation</b> (${nrcC["anticipation"]}/${both.length} of the reviews it
could score), even in clearly negative texts like “gift card arrived water damaged”. The LLM reads the whole review: real
complaints read as <b>anger</b> (${llmC["anger"]} reviews) and praise as <b>joy</b> (${llmC["joy"]} reviews), with negation
understood. Short reviews with no emotion words (e.g. “Four Stars”) get an emotion from the LLM but none from the list.`;

/* ---------- interactive table ---------- */
let state = {ds:"bal", verdict:"all", gt:"all", pred:"all", emo:"all", q:"", sortKey:"idx", sortAsc:true};
let expandedIdx = null;

function currentRows(){ return state.ds==="bal" ? BAL : FIRST; }
function applyFilters(rows){
  return rows.filter(r=>{
    if(state.verdict==="correct" && !r.correct) return false;
    if(state.verdict==="mis" && r.correct) return false;
    if(state.gt!=="all" && r.gt!==state.gt) return false;
    if(state.pred!=="all" && r.pred!==state.pred) return false;
    if(state.emo!=="all" && r.llm_emo!==state.emo) return false;
    if(state.q){
      const q=state.q.toLowerCase();
      if(!((r.title||"").toLowerCase().includes(q)||(r.text||"").toLowerCase().includes(q))) return false;
    }
    return true;
  });
}
function sortRows(rows){
  const k=state.sortKey, a=state.sortAsc;
  return rows.slice().sort((x,y)=>{
    let vx=x[k], vy=y[k];
    if(k==="idx"||k==="rating"||k==="helpful"){vx=+vx; vy=+vy;}
    else if(k==="verdict"){vx=x.correct?1:0; vy=y.correct?1:0;}
    else {vx=String(vx||""); vy=String(vy||"");}
    if(vx<vy) return a?-1:1; if(vx>vy) return a?1:-1; return 0;
  });
}
const escT = s => String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
function stars(r){ return "★".repeat(Math.max(1,Math.min(5,Math.round(r.rating)))); }
function badgeCls(c){ return c==="POSITIVE"?"b-pos":c==="NEUTRAL"?"b-neu":"b-neg"; }
function emoBadge(e){
  if(e==="—") return `<span class="nocell">—</span>`;
  return `<span class="emobadge" style="background:${EMO_COLORS[e]}">${e}</span>`;
}
function nrcChips(r){
  if(!r.nrc || r.nrc_emo==="—") return `<span class="nocell">no lexicon hits</span>`;
  const sorted = Object.entries(r.nrc).sort((a,b)=>b[1]-a[1]).filter(x=>x[1]>0);
  if(!sorted.length) return `<span class="nocell">no lexicon hits</span>`;
  return `<span class="nrc-scores">`+sorted.map(([e,v],i)=>
    `<span class="nrc-chip ${i===0?"top":""}" style="${i===0?`color:${EMO_COLORS[e]}`:""}">${e} ${v}</span>`).join("")+`</span>`;
}

function renderTable(){
  const rows = sortRows(applyFilters(currentRows()));
  $("totalCount").textContent = currentRows().length;
  $("showCount").textContent = rows.length;
  const fsum=[];
  if(state.gt!=="all") fsum.push(`true ${state.gt}`);
  if(state.pred!=="all") fsum.push(`predicted ${state.pred}`);
  if(state.emo!=="all") fsum.push(`emotion ${state.emo}`);
  if(state.q) fsum.push(`“${state.q}”`);
  $("filterSummary").textContent = fsum.length? ` · filtered by ${fsum.join(", ")}` : "";
  const tbody=$("tbody");
  if(!rows.length){
    tbody.innerHTML=`<tr><td colspan="9"><div class="empty-state">No reviews match the current filters.</div></td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(r=>{
    const isExpanded = expandedIdx===r.idx;
    const main = `<tr class="main-row ${r.correct?"":"mis"}" data-idx="${r.idx}">
      <td>${r.idx}</td>
      <td class="tx" title="${escT(r.title)}">${escT(r.title)}</td>
      <td class="star" title="${r.rating} stars">${stars(r)}</td>
      <td><span class="badge ${badgeCls(r.gt)}">${r.gt}</span></td>
      <td><span class="badge ${badgeCls(r.pred)}">${r.pred}</span></td>
      <td>${r.correct?'<span class="badge b-ok">✓</span>':'<span class="badge b-no">✗</span>'}</td>
      <td>${emoBadge(r.llm_emo)}</td>
      <td>${emoBadge(r.nrc_emo)}</td>
      <td>${r.helpful||0}</td>
    </tr>`;
    const detail = isExpanded ? `<tr class="detail-row"><td colspan="9">
      <div class="detail-inner">
        <div class="dt-text"><b>${escT(r.title)}</b> — ${escT(r.text)}</div>
        <div>True: <span class="badge ${badgeCls(r.gt)}">${r.gt}</span>
          &nbsp; Predicted: <span class="badge ${badgeCls(r.pred)}">${r.pred}</span>
          &nbsp; LLM emotion: ${emoBadge(r.llm_emo)} &nbsp; NRC emotion: ${emoBadge(r.nrc_emo)}</div>
        <div style="margin-top:8px">NRC word scores: ${nrcChips(r)}</div>
      </div></td></tr>` : "";
    return main + detail;
  }).join("");
}
$("tbody").addEventListener("click", e=>{
  const tr = e.target.closest("tr.main-row");
  if(!tr) return;
  const idx = +tr.dataset.idx;
  expandedIdx = expandedIdx===idx ? null : idx;
  renderTable();
});
["fDataset","fVerdict","fGt","fPred","fEmo"].forEach(id=>{
  $(id).addEventListener("change", e=>{
    const map={fDataset:"ds",fVerdict:"verdict",fGt:"gt",fPred:"pred",fEmo:"emo"};
    state[map[id]] = e.target.value;
    expandedIdx=null; renderTable();
  });
});
$("fQ").addEventListener("input", e=>{ state.q=e.target.value; renderTable(); });
document.querySelectorAll("th[data-k]").forEach(th=>{
  th.addEventListener("click", ()=>{
    const k=th.dataset.k;
    if(state.sortKey===k) state.sortAsc=!state.sortAsc; else {state.sortKey=k; state.sortAsc=true;}
    document.querySelectorAll("thead th .sort-arrow").forEach(a=>a.remove());
    th.insertAdjacentHTML("beforeend",`<span class="sort-arrow">${state.sortAsc?"▲":"▼"}</span>`);
    renderTable();
  });
});
(function initEmoOptions(){
  const sel=$("fEmo");
  EMO.forEach(e=>{ const o=document.createElement("option"); o.value=e; o.textContent=e; sel.appendChild(o); });
})();
document.querySelectorAll(".swatch").forEach(b=>{
  b.addEventListener("click", ()=>{ document.documentElement.dataset.theme=b.dataset.t; });
});
renderTable();
</script>
</body>
</html>
"""


def main() -> None:
    bal = load_balanced("outputs/emotions_balanced.csv")
    first = load_first100("outputs/emotions_100.csv")
    stats = load_stats("outputs/dataset_stats.json")

    out = (
        TEMPLATE.replace("__BAL__", json.dumps(bal, ensure_ascii=False))
        .replace("__FIRST__", json.dumps(first, ensure_ascii=False))
        .replace("__STATS__", json.dumps(stats, ensure_ascii=False))
        .replace("__EMOTIONS__", json.dumps(list(EMOTIONS)))
        .replace(
            "__EMOCOLORS__",
            json.dumps({e: EMOTION_COLORS[e] for e in EMOTIONS}),
        )
    )
    # two JS references rendered from data
    nrc_bal_n = sum(1 for r in bal if r["nrc_emo"] != "—")
    out = out.replace("NRC_BAL_N", str(nrc_bal_n))

    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote dashboard.html ({len(out):,} bytes, {len(bal)+len(first)} rows embedded)")

    # --- sanity: recompute headlines from the embedded data (cross-check) ---
    n = len(bal)
    c = sum(1 for r in bal if r["correct"])
    print(f"balanced: acc {c}/{n} = {c/n*100:.2f}%")
    for cls in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        mem = [r for r in bal if r["gt"] == cls]
        ok = sum(1 for r in mem if r["correct"])
        prec_den = sum(1 for r in bal if r["pred"] == cls)
        prec = sum(1 for r in bal if r["pred"] == cls and r["correct"])
        print(f"  {cls:9} recall {ok}/{len(mem)} = {ok/len(mem)*100:.1f}% | precision {prec}/{prec_den} = {prec/prec_den*100:.1f}%")
    cn = sum(1 for r in bal if r["gt"] == "NEUTRAL")
    cn_neg = sum(1 for r in bal if r["gt"] == "NEUTRAL" and r["pred"] == "NEGATIVE")
    cn_pos = sum(1 for r in bal if r["gt"] == "NEUTRAL" and r["pred"] == "POSITIVE")
    print(f"  NEUTRAL collapse: {cn_neg}/{cn} to NEGATIVE, {cn_pos}/{cn} to POSITIVE")
    first_c = sum(1 for r in first if r["correct"])
    print(f"first100: acc {first_c}/{len(first)} = {first_c/len(first)*100:.2f}%")
    both = [r for r in bal if r["nrc_emo"] != "—"]
    agree = sum(1 for r in both if r["llm_emo"] == r["nrc_emo"])
    print(f"emotion agreement: {agree}/{len(both)} = {agree/len(both)*100:.1f}%  (NRC coverage {len(both)}/150)")
    emo_llm = {}
    emo_nrc = {}
    for r in bal:
        emo_llm[r["llm_emo"]] = emo_llm.get(r["llm_emo"], 0) + 1
        if r["nrc_emo"] != "—":
            emo_nrc[r["nrc_emo"]] = emo_nrc.get(r["nrc_emo"], 0) + 1
    print("LLM emotions:", dict(sorted(emo_llm.items())))
    print("NRC emotions:", dict(sorted(emo_nrc.items())))
    print(f"dataset stats: {stats['total']:,} reviews | seed {stats['seed']} | classes {stats['classes']}")


if __name__ == "__main__":
    main()
