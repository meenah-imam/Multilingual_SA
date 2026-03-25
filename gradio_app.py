import gradio as gr
import torch
import os
import re
import zipfile
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./naijasenti_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("Model loaded!")

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

LANG_VOCAB = {
    "Hausa": ["kai","na","da","ya","ba","yana","ne","ce","suna","yake","haka","wanda","amma","kuma","shi","su","mu","ku","ni","mai"],
    "Yoruba": ["ti","ni","ko","ati","mo","je","mi","lo","wa","se","bi","pe","si","fun","yi","naa","o","awon","ilu","ile"],
    "Igbo": ["na","ya","ka","gi","nke","bu","ndi","ebe","oge","ihe","obi","ha","m","anyi","noo","gwa","bia","aga","mere","maka"],
    "Pidgin": ["dey","no","dem","wey","make","sef","abeg","wahala","na","wetin","oya","shey","comot","chop","pikin","waka","una","sabi","gbege","jare"],
}

def detect_language(text):
    words = re.findall(r'\b\w+\b', text.lower())
    scores = {lang: sum(1 for w in words if w in vocab) for lang, vocab in LANG_VOCAB.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "English / Unknown"

STOPWORDS = set([
    "the","a","an","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","can",
    "to","of","in","for","on","with","at","by","from","as","it","its","this","that",
    "and","or","but","not","no","so","if","then","than","when","where","who","which",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","our",
    "his","their","what","how","all","just","very","more","also","get","got",
    "na","de","dey","am","im","rt","via","re","da","di","dem","una","sef","sha"
])

def extract_themes(texts, top_n=3):
    words = []
    for t in texts:
        toks = re.findall(r'\b[a-zA-Z]{4,}\b', t.lower())
        words.extend([w for w in toks if w not in STOPWORDS])
    if not words:
        return ["none detected"]
    return [w for w, _ in Counter(words).most_common(top_n)]

session_results = []

def analyze(text):
    if not text.strip():
        return "Enter some text above.", {}, build_summary_html([])

    lang = detect_language(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    idx = int(torch.argmax(logits, dim=-1))
    sentiment = LABEL_MAP[idx]
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    confidence = {
        "Negative": round(probs[0] * 100, 2),
        "Neutral":  round(probs[1] * 100, 2),
        "Positive": round(probs[2] * 100, 2)
    }
    session_results.append({"text": text, "sentiment": sentiment, "lang": lang})
    return sentiment + "  |  " + lang, confidence, build_summary_html(session_results)

def build_summary_html(results):
    if not results:
        return "<div class='summary-empty'>Analyse tweets above to see session summary.</div>"

    total = len(results)
    counts = Counter(r["sentiment"] for r in results)
    pos_pct = round(counts.get("Positive", 0) / total * 100)
    neg_pct = round(counts.get("Negative", 0) / total * 100)
    neu_pct = round(counts.get("Neutral",  0) / total * 100)

    pos_texts = [r["text"] for r in results if r["sentiment"] == "Positive"]
    neg_texts = [r["text"] for r in results if r["sentiment"] == "Negative"]
    pos_themes = ", ".join(extract_themes(pos_texts)) if pos_texts else "none yet"
    neg_themes = ", ".join(extract_themes(neg_texts)) if neg_texts else "none yet"
    langs = Counter(r["lang"] for r in results)
    lang_str = ", ".join(f"{l} ({c})" for l, c in langs.most_common())

    return f"""
    <div class='summary-wrap'>
        <div class='summary-title'>SESSION SUMMARY</div>
        <div class='summary-bars'>
            <div class='bar-row pos'>
                <span class='bar-label'>POSITIVE</span>
                <div class='bar-track'><div class='bar-fill pos-fill' style='width:{pos_pct}%'></div></div>
                <span class='bar-pct'>{pos_pct}%</span>
            </div>
            <div class='bar-row neg'>
                <span class='bar-label'>NEGATIVE</span>
                <div class='bar-track'><div class='bar-fill neg-fill' style='width:{neg_pct}%'></div></div>
                <span class='bar-pct'>{neg_pct}%</span>
            </div>
            <div class='bar-row neu'>
                <span class='bar-label'>NEUTRAL</span>
                <div class='bar-track'><div class='bar-fill neu-fill' style='width:{neu_pct}%'></div></div>
                <span class='bar-pct'>{neu_pct}%</span>
            </div>
        </div>
        <div class='summary-stats'>
            <div class='stat-pill'><span class='stat-num'>{total}</span><span class='stat-lbl'>Total</span></div>
            <div class='stat-pill'><span class='stat-num'>{counts.get("Positive",0)}</span><span class='stat-lbl'>Positive</span></div>
            <div class='stat-pill'><span class='stat-num'>{counts.get("Negative",0)}</span><span class='stat-lbl'>Negative</span></div>
            <div class='stat-pill'><span class='stat-num'>{counts.get("Neutral",0)}</span><span class='stat-lbl'>Neutral</span></div>
        </div>
        <div class='theme-section'>
            <div class='theme-row pos-theme'>
                <span class='theme-icon'>+</span>
                <div><div class='theme-title'>Positive Themes</div><div class='theme-words'>{pos_themes}</div></div>
            </div>
            <div class='theme-row neg-theme'>
                <span class='theme-icon'>-</span>
                <div><div class='theme-title'>Negative Themes</div><div class='theme-words'>{neg_themes}</div></div>
            </div>
        </div>
        <div class='lang-row'>
            <span class='lang-label'>Languages:</span>
            <span class='lang-val'>{lang_str}</span>
        </div>
    </div>
    """

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body {
    height: 100%;
    overflow: hidden !important;
    margin: 0;
    padding: 0;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0d0221 0%, #1a0533 30%, #0a1628 60%, #001a1a 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    height: 100vh !important;
    overflow: hidden !important;
    padding: 0.3rem 0.9rem !important;
    margin: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    zoom: 0.75 !important;
    transform-origin: top left !important;
}

.contain { height: 100% !important; display: flex !important; flex-direction: column !important; }
.gap { gap: 0.5rem !important; }

/* Header */
#header {
    text-align: center;
    padding: 0.4rem 0 0.3rem;
    flex-shrink: 0;
}

#header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ff6ec7, #ffb347, #00f5a0, #00d9f5, #ff6ec7) !important;
    background-size: 300% !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: gradientShift 5s ease infinite !important;
    margin: 0 !important;
    line-height: 1.1 !important;
}

#header p {
    color: #c4a8e0 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    margin: 0.15rem 0 0 !important;
}

/* Main row */
.main-row {
    display: flex !important;
    gap: 0.8rem !important;
    flex: 1 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

/* Left column */
.left-col {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.5rem !important;
    flex: 3 !important;
    min-height: 0 !important;
}

/* Right column */
.right-col {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.5rem !important;
    flex: 2 !important;
    min-height: 0 !important;
}

/* Tweet input */
#tweet-box {
    flex-shrink: 0 !important;
}

#tweet-box textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 2px solid rgba(255,110,199,0.35) !important;
    border-radius: 12px !important;
    color: #000000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 0.9rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    resize: none !important;
    min-height: 80px !important;
    max-height: 90px !important;
}

#tweet-box textarea:focus {
    border-color: #ff6ec7 !important;
    box-shadow: 0 0 20px rgba(255,110,199,0.25) !important;
    outline: none !important;
}

#tweet-box textarea::placeholder { color: #8a6aaa !important; font-weight: 500 !important; }

#tweet-box label {
    color: #ff6ec7 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

/* Button */
#analyze-btn button {
    background: linear-gradient(135deg, #ff6ec7, #ff8c42) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.3s !important;
    box-shadow: 0 3px 18px rgba(255,110,199,0.45) !important;
    width: 100% !important;
}

#analyze-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(255,110,199,0.65) !important;
}

/* Summary panel */
#summary-out {
    flex: 1 !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

.summary-empty {
    text-align: center;
    color: #6b5a8a;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 1.5rem;
    border: 2px dashed rgba(255,110,199,0.2);
    border-radius: 12px;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.summary-wrap {
    background: rgba(255,255,255,0.04);
    border: 2px solid rgba(255,110,199,0.2);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    height: 100%;
    font-family: 'DM Sans', sans-serif;
}

.summary-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem;
    font-weight: 800;
    letter-spacing: 0.2em;
    color: #ff6ec7;
    margin-bottom: 0.7rem;
}

.summary-bars { display: flex; flex-direction: column; gap: 0.45rem; margin-bottom: 0.8rem; }

.bar-row { display: flex; align-items: center; gap: 0.6rem; }
.bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    width: 60px;
    flex-shrink: 0;
}
.bar-row.pos .bar-label { color: #00f5a0; }
.bar-row.neg .bar-label { color: #ff4d6d; }
.bar-row.neu .bar-label { color: #f5a623; }

.bar-track { flex:1; height:6px; background:rgba(255,255,255,0.08); border-radius:100px; overflow:hidden; }
.bar-fill { height:100%; border-radius:100px; }
.pos-fill { background: linear-gradient(90deg,#00f5a0,#00d9f5); }
.neg-fill { background: linear-gradient(90deg,#ff4d6d,#ff8c42); }
.neu-fill { background: linear-gradient(90deg,#f5a623,#ffe066); }

.bar-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #fff;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

.summary-stats {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 0.4rem;
    margin-bottom: 0.8rem;
}

.stat-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.45rem 0.3rem;
    text-align: center;
}

.stat-num {
    display: block;
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}

.stat-lbl {
    display: block;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #7a6090;
    margin-top: 0.15rem;
}

.theme-section { display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 0.6rem; }

.theme-row {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.45rem 0.7rem;
    border-radius: 8px;
}
.theme-row.pos-theme { background:rgba(0,245,160,0.08); border:1px solid rgba(0,245,160,0.2); }
.theme-row.neg-theme { background:rgba(255,77,109,0.08); border:1px solid rgba(255,77,109,0.2); }

.theme-icon {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    line-height: 1.3;
    flex-shrink: 0;
}
.pos-theme .theme-icon { color: #00f5a0; }
.neg-theme .theme-icon { color: #ff4d6d; }

.theme-title { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #8a7aaa; margin-bottom: 0.1rem; }
.theme-words { font-size: 0.8rem; font-weight: 700; color: #e8e0f4; }

.lang-row { border-top:1px solid rgba(255,255,255,0.08); padding-top:0.5rem; font-size:0.75rem; }
.lang-label { color:#7a6090; font-weight:700; margin-right:0.4rem; }
.lang-val { color:#d4a8ff; font-weight:700; }

/* Sentiment output */
#sentiment-out {
    flex-shrink: 0 !important;
    background: rgba(0,245,160,0.07) !important;
    border: 2px solid rgba(0,245,160,0.3) !important;
    border-radius: 12px !important;
    padding: 0.6rem 0.9rem !important;
}

#sentiment-out label {
    color: #00f5a0 !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

#sentiment-out .output-class, #sentiment-out span {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    color: #00f5a0 !important;
}

/* Confidence output */
#confidence-out {
    flex-shrink: 0 !important;
    background: rgba(0,217,245,0.07) !important;
    border: 2px solid rgba(0,217,245,0.3) !important;
    border-radius: 12px !important;
    padding: 0.6rem 0.9rem !important;
}

#confidence-out label {
    color: #00d9f5 !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}

#confidence-out * {
    background: transparent !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
    font-family: 'DM Sans', sans-serif !important;
    text-shadow: none !important;
    -webkit-font-smoothing: antialiased !important;
}

#confidence-out label {
    color: #00d9f5 !important;
    font-size: 0.62rem !important;
}

/* Footer */
#footer {
    text-align: center;
    color: #a889cc !important;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.4rem 0;
    flex-shrink: 0;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
}

.footer-uni {
    color: #d4a8ff;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
"""

with gr.Blocks(css=custom_css, title="Naija Multilingual Feedback") as demo:

    gr.HTML("""
    <div id="header">
        <h1>Naija Multilingual Feedback</h1>
        <p>Sentiment Analysis across English &bull; Hausa &bull; Igbo &bull; Pidgin &bull; Yoruba</p>
    </div>
    """)

    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=3, elem_classes="left-col"):
            text_input = gr.Textbox(
                placeholder="Type or paste a Nigerian tweet here...",
                label="Tweet",
                lines=3,
                max_lines=4,
                elem_id="tweet-box"
            )
            analyze_btn = gr.Button(
                "Analyse Sentiment",
                variant="primary",
                elem_id="analyze-btn"
            )
            summary_out = gr.HTML(
                value="<div class='summary-empty'>Analyse tweets to see session summary.</div>",
                elem_id="summary-out"
            )

        with gr.Column(scale=2, elem_classes="right-col"):
            sentiment_out = gr.Label(
                label="Detected Sentiment  |  Language",
                elem_id="sentiment-out"
            )
            confidence_out = gr.Label(
                label="Confidence Scores %",
                elem_id="confidence-out"
            )

    gr.HTML('<div id="footer"><span class="footer-uni">Iconic University &mdash; Capstone Project</span> &nbsp;&bull;&nbsp; XLM-ROBERTA &bull; NAIJASENTI DATASET</div>')

    analyze_btn.click(
        fn=analyze,
        inputs=text_input,
        outputs=[sentiment_out, confidence_out, summary_out]
    )
    text_input.submit(
        fn=analyze,
        inputs=text_input,
        outputs=[sentiment_out, confidence_out, summary_out]
    )

demo.launch(share=True)
