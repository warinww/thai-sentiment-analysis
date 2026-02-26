
# =============================================================================================
# 0) Install + Imports
# =============================================================================================

import json
import numpy as np
import pandas as pd

from pythainlp.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

print("TF:", tf.__version__)


# =============================================================================================
# 1) Upload cleaned JSON
# =============================================================================================

with open("train_sentiment.cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = data["text"]
sentiments = data["sentiment"]

rows = []
for k, t in texts.items():
    if k in sentiments:
        rows.append((k, t, sentiments[k]))

df = pd.DataFrame(rows, columns=["id", "text", "sentiment"])
print(df.shape)
df.head()

# ====================================================================================================
# 2) Tokenize Thai (word-level)
# ====================================================================================================
def tokenize_th(s: str):
    # ใช้ newmm เสถียร + เก็บ token พวก <DATE> <POS_EMO> ได้เป็นคำ
    toks = word_tokenize(str(s), engine="newmm", keep_whitespace=False)
    return toks

df["tokens"] = df["text"].map(tokenize_th)
df["text_tok"] = df["tokens"].map(lambda toks: " ".join(toks))
df[["text", "text_tok", "sentiment"]].head(3)


# ======================================================================================================
# 3) Encode labels
# ======================================================================================================
label2id = {"negative": 0, "neutral": 1, "positive": 2}
df["y"] = df["sentiment"].map(label2id)

# ตรวจว่ามี label หลุดไหม
if df["y"].isna().any():
    bad = df[df["y"].isna()]["sentiment"].value_counts()
    raise ValueError(f"Found unknown labels:\n{bad}")

print(df["y"].value_counts())


# ===============================================================================================================
# 4) Split 70/15/15 (stratified)
# ===============================================================================================================
X = df["text_tok"].values
y = df["y"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("train:", len(X_train), "val:", len(X_val), "test:", len(X_test))



# ===============================================================================================================
# 4) Split 70/15/15 (stratified)
# ===============================================================================================================
X = df["text_tok"].values
y = df["y"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("train:", len(X_train), "val:", len(X_val), "test:", len(X_test))


# ---------------------------------------------------------------------------------------------------------------------------------------

# ============================================================
# STACKING (SVM + LR + NB + GRADIENT BOOST)
# ============================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
#  1 TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=80000,
    ngram_range=(1,3),
    min_df=2,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

n_classes = len(np.unique(y_train))

# ============================================================
# 2 SVD for Gradient Boosting
# ============================================================

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_vec)
X_test_svd  = svd.transform(X_test_vec)

scaler_svd = StandardScaler()
X_train_svd = scaler_svd.fit_transform(X_train_svd)
X_test_svd  = scaler_svd.transform(X_test_svd)

# เพิ่มอันนี้
X_val_vec = vectorizer.transform(X_val)

# สำหรับ SVD
X_val_svd = svd.transform(X_val_vec)
X_val_svd = scaler_svd.transform(X_val_svd)

# ============================================================
#  3 Base Models
# ============================================================

svm = CalibratedClassifierCV(
    LinearSVC(C=1.5),
    method="sigmoid",
    cv=3
)

lr = LogisticRegression(max_iter=3000)
nb = MultinomialNB(alpha=0.5)

gb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    max_iter=200
)

base_models_sparse = [svm, lr, nb]
base_models_dense  = [gb]

# ============================================================
#  4 OOF Stacking
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

total_models = len(base_models_sparse) + len(base_models_dense)
oof_train = np.zeros((X_train_vec.shape[0], n_classes * total_models))
oof_test  = np.zeros((X_test_vec.shape[0], n_classes * total_models))

model_index = 0

# ---- Sparse models (SVM, LR, NB)
for model in base_models_sparse:

    oof_test_fold = np.zeros((X_test_vec.shape[0], n_classes, 5))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_vec, y_train)):

        X_tr, X_val_fold = X_train_vec[train_idx], X_train_vec[val_idx]
        y_tr = y_train[train_idx]

        model.fit(X_tr, y_tr)

        val_proba = model.predict_proba(X_val_fold)
        test_proba = model.predict_proba(X_test_vec)

        oof_train[val_idx, model_index*n_classes:(model_index+1)*n_classes] = val_proba
        oof_test_fold[:, :, fold] = test_proba

    oof_test[:, model_index*n_classes:(model_index+1)*n_classes] = oof_test_fold.mean(axis=2)
    model_index += 1


# ---- Dense model (Gradient Boosting)
for model in base_models_dense:

    oof_test_fold = np.zeros((X_test_svd.shape[0], n_classes, 5))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_svd, y_train)):

        X_tr, X_val_fold = X_train_svd[train_idx], X_train_svd[val_idx]
        y_tr = y_train[train_idx]

        model.fit(X_tr, y_tr)

        val_proba = model.predict_proba(X_val_fold)
        test_proba = model.predict_proba(X_test_svd)

        oof_train[val_idx, model_index*n_classes:(model_index+1)*n_classes] = val_proba
        oof_test_fold[:, :, fold] = test_proba

    oof_test[:, model_index*n_classes:(model_index+1)*n_classes] = oof_test_fold.mean(axis=2)
    model_index += 1


# ============================================================
#  5 Meta Learner
# ============================================================

meta_model = LogisticRegression(max_iter=3000)
meta_model.fit(oof_train, y_train)

train_pred = meta_model.predict(oof_train)
test_pred  = meta_model.predict(oof_test)

print("\n🔥 STACKING RESULTS (WITH GRADIENT BOOST)")
print("Train Acc:", accuracy_score(y_train, train_pred))
print("Test Acc :", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred, digits=4))
print(confusion_matrix(y_test, test_pred))

# ============================================================
# 🔎 BASE MODEL TRAIN ACC
# ============================================================

print("\n--- Base Model Accuracy (Full Train Fit) ---")

# ----- Sparse Models
for model in base_models_sparse:
    model.fit(X_train_vec, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_vec))
    val_acc   = accuracy_score(y_val, model.predict(X_val_vec))
    test_acc  = accuracy_score(y_test, model.predict(X_test_vec))

    print(f"{model.__class__.__name__}: "
          f"Train={train_acc:.4f} | "
          f"Val={val_acc:.4f} | "
          f"Test={test_acc:.4f}")

# ----- Dense Model (Gradient Boost)
for model in base_models_dense:
    model.fit(X_train_svd, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_svd))
    val_acc   = accuracy_score(y_val, model.predict(X_val_svd))
    test_acc  = accuracy_score(y_test, model.predict(X_test_svd))

    print(f"{model.__class__.__name__}: "
          f"Train={train_acc:.4f} | "
          f"Val={val_acc:.4f} | "
          f"Test={test_acc:.4f}")

    # สร้าง meta feature ของ val set

meta_val = np.zeros((X_val_vec.shape[0], n_classes * total_models))
model_index = 0

# sparse models
for model in base_models_sparse:
    model.fit(X_train_vec, y_train)
    val_proba = model.predict_proba(X_val_vec)

    meta_val[:, model_index*n_classes:(model_index+1)*n_classes] = val_proba
    model_index += 1

# dense model
for model in base_models_dense:
    model.fit(X_train_svd, y_train)
    val_proba = model.predict_proba(X_val_svd)

    meta_val[:, model_index*n_classes:(model_index+1)*n_classes] = val_proba
    model_index += 1

# predict
val_pred = meta_model.predict(meta_val)

print("\n🔥 STACKING VAL ACC:", accuracy_score(y_val, val_pred))

print("\n🔥 Validated RESULTS")
print(classification_report(y_val, val_pred, digits=4))
print(confusion_matrix(y_val, val_pred))

# ============================================================
# DOwnload Model
# ============================================================

# retrain base models on full train
for model in base_models_sparse:
    model.fit(X_train_vec, y_train)

for model in base_models_dense:
    model.fit(X_train_svd, y_train)

import joblib

joblib.dump({
    "vectorizer": vectorizer,
    "svd": svd,
    "scaler_svd": scaler_svd,
    "base_models_sparse": base_models_sparse,
    "base_models_dense": base_models_dense,
    "meta_model": meta_model,
    "n_classes": n_classes
}, "sentiment_stacking.pkl")

print("Saved successfully")

from google.colab import files
files.download("sentiment_stacking.pkl")
