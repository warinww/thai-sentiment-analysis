import gradio as gr
import joblib
import numpy as np
import json
import tempfile
import os
from pythainlp.tokenize import word_tokenize

# ---- Load model ----
LABELS = ["negative", "neutral", "positive"]

model_path = "sentiment_stacking.pkl"  # แก้ path ตามที่เก็บไฟล์
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
    toks = word_tokenize(text, engine="newmm", keep_whitespace=False)

    if USE_BIGRAM:
        toks = add_bigrams(toks)

    return " ".join(toks)

def predict(text: str):
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

    prob_dict = {LABELS[i]: float(prob[i]) for i in range(len(LABELS))}
    return pred, prob_dict, text_tok

def predict_json(file):
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

    output_path = "predicted_output.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)

    return output_path

# ---- UI ----
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

file_demo = gr.Interface(
    fn=predict_json,
    inputs=gr.File(label="Upload JSON file"),
    outputs=gr.File(label="Download result JSON"),
    title="Predict JSON File"
)

demo = gr.TabbedInterface(
    [text_demo, file_demo],
    tab_names=["Text Input", "JSON Upload"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))