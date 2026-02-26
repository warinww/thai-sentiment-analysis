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

def clean_json_keep_sentiment(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data["text"]
    sentiments = data["sentiment"]

    cleaned_texts = {}
    cleaned_sentiments = {}

    for k in texts:
        if k in sentiments:
            cleaned_texts[k] = clean_text(texts[k])
            cleaned_sentiments[k] = sentiments[k]

    out = {
        "text": cleaned_texts,
        "sentiment": cleaned_sentiments
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved ->", output_path)
    print("Total rows:", len(cleaned_texts))


# -------- RUN --------
if __name__ == "__main__":
    INPUT_JSON = "train_sentiment.json"
    OUTPUT_JSON = "train_sentiment.cleaned.json"
    clean_json_keep_sentiment(INPUT_JSON, OUTPUT_JSON)