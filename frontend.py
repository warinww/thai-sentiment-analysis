import json
import re
import html
import unicodedata
import regex as reg  # pip install regex

# ---------- Regex ----------
URL_RE = re.compile(r"""(?i)\b((?:https?://|www\.)[^\s<>"]+|[a-z0-9.-]+\.(?:com|net|org|co|me|io|info|tv|th)(?:/[^\s<>"]*)?)""")
MENTION_RE = re.compile(r"(?<!\w)@[\w\.\-ก-๙_]+")
HASHTAG_RE = re.compile(r"(?<!\w)#[\wก-๙_]+")
TIME_RE = re.compile(r"\b\d{1,2}\s*[:\.]\s*\d{2}\s*(?:น\.|น|am|pm)?", re.IGNORECASE)

TH_MONTHS = r"(?:ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.|มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม)"
DATE_RE = re.compile(
    rf"\b\d{{1,2}}\s*(?:{TH_MONTHS})\s*(?:\d{{2,4}})\b|\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b",
    re.IGNORECASE
)

YEAR_RE = re.compile(r"\bพ\.ศ\.?\s*\d{4}\b|\b20\d{2}\b|\b25\d{2}\b")
NUM_RE  = re.compile(r"\b\d+(?:[,\.\:]\d+)*\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
DOT_PATTERN_RE = re.compile(r"\s*\.\s*(?:\.\s*)+")
LAUGH_TEXT = re.compile(r"\b5{3,}\b")

# ---- emoji (ครอบคลุมของตกแต่งทั้งหมด)
EMOJI_SEQ_RE = reg.compile(
    r"(?:\p{Extended_Pictographic}(?:\uFE0F|\uFE0E)?(?:\u200D\p{Extended_Pictographic}(?:\uFE0F|\uFE0E)?)*)"
)
EMOJI_TRAIL_RE = reg.compile(r"[\u200D\uFE0F\uFE0E]")

# ---- Emoji mapping
POS = ["😂","🤣","😆","😄","😊","🙂","😍","🥰","😘","❤️","💖","💕","👍","👏","😁","😃"]
NEG = ["😡","😠","🤬","😭","😢","😞","😤","😱","🤢","🤮","💔","👎"]
SHOCK = ["😲","😮","😳","😨","😰"]

def normalize_repeats(s: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", s)

def emoji_to_tags_then_drop_rest(text: str) -> str:
    for e in POS:
        text = text.replace(e, " <POS_EMO> ")
    for e in NEG:
        text = text.replace(e, " <NEG_EMO> ")
    for e in SHOCK:
        text = text.replace(e, " <SHOCK> ")

    text = LAUGH_TEXT.sub(" <LAUGH> ", text)

    # ลบ emoji ที่เหลือทั้งหมด
    text = EMOJI_SEQ_RE.sub(" ", text)
    text = EMOJI_TRAIL_RE.sub(" ", text)

    return text

def clean_text(s: str) -> str:
    if not s:
        return "<EMPTY>"

    s = html.unescape(s)
    s = unicodedata.normalize("NFC", s)

    s = HTML_TAG_RE.sub(" ", s)
    s = URL_RE.sub(" <URL> ", s)
    s = MENTION_RE.sub(" <USER> ", s)
    s = HASHTAG_RE.sub(lambda m: " " + m.group(0)[1:] + " ", s)

    s = emoji_to_tags_then_drop_rest(s)

    s = DOT_PATTERN_RE.sub(" ", s)
    s = re.sub(r"\s\.\s", " ", s)

    s = DATE_RE.sub(" <DATE> ", s)
    s = TIME_RE.sub(" <TIME> ", s)
    s = YEAR_RE.sub(" <YEAR> ", s)
    s = NUM_RE.sub(" <NUM> ", s)

    s = normalize_repeats(s)
    s = re.sub(r"\s+", " ", s).strip()

    return s if len(s) >= 2 else "<EMPTY>"

!pip -q install gradio pythainlp joblib

import gradio as gr
import joblib
import numpy as np
import json
import tempfile
import os
from pythainlp.tokenize import word_tokenize

LABELS = ["negative", "neutral", "positive"]

# โหลดโมเดล
from google.colab import drive
drive.mount('/content/drive')

model_path = "/content/drive/MyDrive/sentiment_stacking.pkl"
data = joblib.load(model_path)

vectorizer = data["vectorizer"]
svd = data["svd"]
scaler_svd = data["scaler_svd"]
base_models_sparse = data["base_models_sparse"]
base_models_dense = data["base_models_dense"]
meta_model = data["meta_model"]

USE_BIGRAM = True

def add_bigrams(tokens):
    if len(tokens) < 2:
        return tokens
    bigrams = [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

def preprocess(text: str) -> str:
    text = clean_text(text)   # ใช้ clean ก่อน

    if text == "<EMPTY>":
        return ""

    toks = word_tokenize(text, engine="newmm", keep_whitespace=False)

    if USE_BIGRAM:
        toks = add_bigrams(toks)

    return " ".join(toks)

def predict(text: str):

    text_tok = preprocess(text)
    X_vec = vectorizer.transform([text_tok])

    meta_features = []

    # sparse models
    for model in base_models_sparse:
        meta_features.append(model.predict_proba(X_vec))

    # dense model
    X_svd = scaler_svd.transform(svd.transform(X_vec))
    for model in base_models_dense:
        meta_features.append(model.predict_proba(X_svd))

    meta_features = np.hstack(meta_features)

    prob = meta_model.predict_proba(meta_features)[0]
    pred = LABELS[int(np.argmax(prob))]

    prob_dict = {LABELS[i]: float(prob[i]) for i in range(len(LABELS))}
    return pred, prob_dict, text_tok

def predict_json(file):

    # ชื่อไฟล์เดิม
    original_name = os.path.basename(file.name)
    base_name = os.path.splitext(original_name)[0]

    with open(file.name, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    texts = data_json["text"]
    sentiments = {}

    for k, text in texts.items():

        text_tok = preprocess(text)
        X_vec = vectorizer.transform([text_tok])

        meta_features = []

        for model in base_models_sparse:
            meta_features.append(model.predict_proba(X_vec))

        X_svd = scaler_svd.transform(svd.transform(X_vec))
        for model in base_models_dense:
            meta_features.append(model.predict_proba(X_svd))

        meta_features = np.hstack(meta_features)

        prob = meta_model.predict_proba(meta_features)[0]
        pred = LABELS[int(np.argmax(prob))]

        sentiments[k] = pred

    data_json["sentiment"] = sentiments

    # สร้างชื่อไฟล์ใหม่
    output_filename = f"{base_name}_predicted.json"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    return output_path

# -------- Tab 1: Predict single text --------
text_demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Text"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Label(label="Probabilities"),
        gr.Textbox(lines=2, label="After preprocess")
    ],
    title="Predict Single Text"
)

# -------- Tab 2: Predict JSON file --------
file_demo = gr.Interface(
    fn=predict_json,
    inputs=gr.File(label="Upload JSON file"),
    outputs=gr.File(label="Download result JSON"),
    title="Predict JSON File"
)

# -------- Combine --------
demo = gr.TabbedInterface(
    [text_demo, file_demo],
    tab_names=["Text Input", "JSON Upload"]
)

demo.launch(share=True)