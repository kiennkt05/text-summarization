import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from underthesea import pos_tag, word_tokenize
except ImportError as e:
    raise ImportError(
        "Bạn cần cài underthesea: pip install underthesea"
    ) from e


# =========================
# Utils
# =========================
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s, flags=re.UNICODE)
    return s


def safe_word_count(text: str) -> int:
    if not text:
        return 0
    return len([t for t in text.strip().split() if t])


def f1_score_from_sets(pred: Set[str], ref: Set[str]) -> Tuple[float, float, float]:
    if not pred and not ref:
        # Không có thực thể nào → không thể đánh giá → trả về 0 để tránh điểm ảo
        return 0.0, 0.0, 0.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not ref:
        return 0.0, 0.0, 0.0

    inter = len(pred & ref)
    precision = inter / len(pred) if pred else 0.0
    recall = inter / len(ref) if ref else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def gaussian_score(x: float, target: float = 1.0, sigma: float = 0.35) -> float:
    if x is None or np.isnan(x):
        return 0.0
    return float(math.exp(-0.5 * ((x - target) / sigma) ** 2))


# =========================
# Entity extraction (nguyên bản – dạng set)
# =========================
NUM_PATTERN = re.compile(
    r"""
    (?<!\w)
    (
        \d{1,2}/\d{1,2}(?:/\d{2,4})?      # 19/4, 19/4/2024
        |
        \d{1,2}-\d{1,2}(?:-\d{2,4})?      # 15-17
        |
        \d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)? # 550.000, 95.36
        |
        \d+(?:[.,]\d+)?%?                 # 20, 89, 95,36, 50%
    )
    (?!\w)
    """,
    re.VERBOSE,
)

DATE_PATTERN = re.compile(
    r"""
    (?<!\w)
    (
        \d{1,2}/\d{1,2}(?:/\d{2,4})?
        |
        \d{1,2}-\d{1,2}(?:-\d{2,4})?
        |
        \b\d{4}\b
    )
    (?!\w)
    """,
    re.VERBOSE,
)


def extract_dates(text: str) -> Set[str]:
    if not text:
        return set()
    raw = DATE_PATTERN.findall(text)
    out = set()
    for x in raw:
        x = normalize_text(x)
        if x:
            out.add(x)
    return out


def extract_numbers(text: str) -> Set[str]:
    if not text:
        return set()
    dates = extract_dates(text)
    raw = NUM_PATTERN.findall(text)
    out = set()
    for x in raw:
        x = normalize_text(x)
        if not x:
            continue
        if x in dates:
            continue
        x2 = x.replace("%", "")
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?", x2):
            x2 = x2.replace(".", "").replace(",", "")
        else:
            if "," in x2 and "." not in x2:
                x2 = x2.replace(",", ".")
        x2 = normalize_text(x2)
        if x2:
            out.add(x2)
    return out


def extract_proper_nouns(text: str) -> Set[str]:
    if not text:
        return set()
    try:
        tagged = pos_tag(text)
    except Exception:
        return set()
    phrases = []
    current = []
    for item in tagged:
        if not item:
            continue
        token = item[0]
        tag = item[1] if len(item) > 1 else ""
        if tag == "Np":
            current.append(token)
        else:
            if current:
                phrase = normalize_text(" ".join(current))
                if phrase:
                    phrases.append(phrase)
                current = []
    if current:
        phrase = normalize_text(" ".join(current))
        if phrase:
            phrases.append(phrase)
    phrases = [p for p in phrases if len(p) >= 2]
    return set(phrases)


@dataclass
class EntityBundle:
    proper_nouns: Set[str]
    numbers: Set[str]
    dates: Set[str]


def extract_entity_bundle(text: str) -> EntityBundle:
    return EntityBundle(
        proper_nouns=extract_proper_nouns(text),
        numbers=extract_numbers(text),
        dates=extract_dates(text),
    )


# =========================
# REPETITION PENALTY
# =========================
def distinct_n(text: str, n: int) -> float:
    tokens = word_tokenize(normalize_text(text))
    tokens = [t for t in tokens if t.strip()]
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def repetition_penalty(text: str) -> float:
    """
    Trả về hệ số phạt trong [0, 1].
    Càng lặp nhiều thì giá trị càng nhỏ.
    """
    d2 = distinct_n(text, 2)
    d3 = distinct_n(text, 3)
    distinct_min = min(d2, d3)
    return distinct_min ** 2   # bình phương để phạt mạnh hơn


# =========================
# Các hàm tính điểm cập nhật
# =========================
def category_f1(pred_set: Set[str], ref_set: Set[str]) -> Tuple[float, float, float]:
    return f1_score_from_sets(pred_set, ref_set)


def compute_entity_ref_score(pred_text: str, ref_text: str, args) -> dict:
    pred = extract_entity_bundle(pred_text)
    ref = extract_entity_bundle(ref_text)

    pn_p, pn_r, pn_f1 = category_f1(pred.proper_nouns, ref.proper_nouns)
    num_p, num_r, num_f1 = category_f1(pred.numbers, ref.numbers)
    date_p, date_r, date_f1 = category_f1(pred.dates, ref.dates)

    raw_entity_ref_score = (
        args.proper_noun_weight * pn_f1
        + args.numeric_weight * num_f1
        + args.date_weight * date_f1
    )

    penalty = repetition_penalty(pred_text)
    entity_ref_score = raw_entity_ref_score * penalty

    return {
        "pn_precision_ref": pn_p,
        "pn_recall_ref": pn_r,
        "pn_f1_ref": pn_f1,
        "num_precision_ref": num_p,
        "num_recall_ref": num_r,
        "num_f1_ref": num_f1,
        "date_precision_ref": date_p,
        "date_recall_ref": date_r,
        "date_f1_ref": date_f1,
        "entity_ref_score": entity_ref_score,
    }


def compute_entity_article_precision(pred_text: str, article_text: str, args) -> dict:
    pred = extract_entity_bundle(pred_text)
    art = extract_entity_bundle(article_text)

    pn_p, pn_r, pn_f1 = category_f1(pred.proper_nouns, art.proper_nouns)
    num_p, num_r, num_f1 = category_f1(pred.numbers, art.numbers)
    date_p, date_r, date_f1 = category_f1(pred.dates, art.dates)

    raw_entity_article_precision = (
        args.proper_noun_weight * pn_p
        + args.numeric_weight * num_p
        + args.date_weight * date_p
    )

    penalty = repetition_penalty(pred_text)
    entity_article_precision = raw_entity_article_precision * penalty

    return {
        "pn_precision_article": pn_p,
        "pn_recall_article": pn_r,
        "pn_f1_article": pn_f1,
        "num_precision_article": num_p,
        "num_recall_article": num_r,
        "num_f1_article": num_f1,
        "date_precision_article": date_p,
        "date_recall_article": date_r,
        "date_f1_article": date_f1,
        "entity_article_precision": entity_article_precision,
    }


def compute_compression_score(pred_text: str, ref_text: str, args) -> Tuple[float, float]:
    pred_len = safe_word_count(pred_text)
    ref_len = safe_word_count(ref_text)
    if ref_len == 0:
        return 0.0, 0.0
    ratio = pred_len / ref_len
    score = gaussian_score(ratio, target=args.compression_target, sigma=args.compression_sigma)
    return ratio, score

# =========================
# Main processing
# =========================
def load_data(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Prepare RL dataset")
    parser.add_argument("--input_path", type=str, default="/content/lora_train_results.json", help="Input JSON file")
    parser.add_argument("--full_output_path", type=str, default="full_RL_dataset.parquet", help="Full dataset output Parquet")
    parser.add_argument("--worst_output_path", type=str, default="RL_dataset.parquet", help="Worst K ratio output Parquet")
    
    parser.add_argument("--article_col", type=str, default="article", help="Article column name")
    parser.add_argument("--reference_col", type=str, default="references", help="Reference column name")
    parser.add_argument("--prediction_col", type=str, default="predictions", help="Prediction column name")
    parser.add_argument("--bert_col", type=str, default="bert-f1", help="BERT score column name")
    parser.add_argument("--rouge_col", type=str, default="rouge-L", help="ROUGE score column name")
    
    parser.add_argument("--top_k_ratio", type=float, default=0.25, help="Top K ratio for worst samples")
    
    parser.add_argument("--bert_weight", type=float, default=0.35, help="Weight for BERT score")
    parser.add_argument("--rouge_weight", type=float, default=0.25, help="Weight for ROUGE score")
    parser.add_argument("--entity_weight", type=float, default=0.20, help="Weight for entity score")
    parser.add_argument("--compression_weight", type=float, default=0.20, help="Weight for compression score")
    
    parser.add_argument("--proper_noun_weight", type=float, default=0.40, help="Weight for proper noun score")
    parser.add_argument("--numeric_weight", type=float, default=0.35, help="Weight for numeric score")
    parser.add_argument("--date_weight", type=float, default=0.25, help="Weight for date score")
    
    parser.add_argument("--entity_ref_weight", type=float, default=0.70, help="Weight for entity reference score")
    parser.add_argument("--entity_article_weight", type=float, default=0.30, help="Weight for entity article precision")
    
    parser.add_argument("--compression_target", type=float, default=1.0, help="Target ratio for compression")
    parser.add_argument("--compression_sigma", type=float, default=0.35, help="Sigma for compression Gaussian score")
    
    args = parser.parse_args()

    df = load_data(args.input_path)

    # Chuẩn bị các list
    entity_ref_scores = []
    entity_article_precisions = []
    compression_ratios = []
    compression_scores = []
    rep_penalties = []

    pn_f1_refs = []
    num_f1_refs = []
    date_f1_refs = []
    pn_precision_articles = []
    num_precision_articles = []
    date_precision_articles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring rows"):
        article = "" if pd.isna(row[args.article_col]) else str(row[args.article_col])
        reference = "" if pd.isna(row[args.reference_col]) else str(row[args.reference_col])
        prediction = "" if pd.isna(row[args.prediction_col]) else str(row[args.prediction_col])

        # entity_ref_score và entity_article_precision đã được nhân với repetition_penalty bên trong
        ref_scores = compute_entity_ref_score(prediction, reference, args)
        art_scores = compute_entity_article_precision(prediction, article, args)
        comp_ratio, comp_score = compute_compression_score(prediction, reference, args)
        penalty = repetition_penalty(prediction)

        entity_ref_scores.append(ref_scores["entity_ref_score"])
        entity_article_precisions.append(art_scores["entity_article_precision"])
        compression_ratios.append(comp_ratio)
        compression_scores.append(comp_score)
        rep_penalties.append(penalty)

        pn_f1_refs.append(ref_scores["pn_f1_ref"])
        num_f1_refs.append(ref_scores["num_f1_ref"])
        date_f1_refs.append(ref_scores["date_f1_ref"])
        pn_precision_articles.append(art_scores["pn_precision_article"])
        num_precision_articles.append(art_scores["num_precision_article"])
        date_precision_articles.append(art_scores["date_precision_article"])

    # Gán vào DataFrame
    df["pn_f1_ref"] = pn_f1_refs
    df["num_f1_ref"] = num_f1_refs
    df["date_f1_ref"] = date_f1_refs
    df["entity_ref_score"] = entity_ref_scores

    df["pn_precision_article"] = pn_precision_articles
    df["num_precision_article"] = num_precision_articles
    df["date_precision_article"] = date_precision_articles
    df["entity_article_precision"] = entity_article_precisions

    df["compression_ratio"] = compression_ratios
    df["compression_score"] = compression_scores
    df["repetition_penalty"] = rep_penalties

    # EntityScore tổng hợp (các thành phần đã chứa phạt lặp)
    df["entity_score"] = (
        args.entity_ref_weight * df["entity_ref_score"]
        + args.entity_article_weight * df["entity_article_precision"]
    )

    # FinalScore với trọng số mới
    df["final_score"] = (
        args.bert_weight * df[args.bert_col]
        + args.rouge_weight * df[args.rouge_col]
        + args.entity_weight * df["entity_score"]
        + args.compression_weight * df["compression_score"]
    )

    # Xếp hạng: score thấp → mẫu tệ
    df = df.sort_values("final_score", ascending=True).reset_index(drop=True)
    df["rank_worst_first"] = np.arange(1, len(df) + 1)

    # Lấy 25% mẫu tệ nhất (hoặc tỷ lệ tùy chỉnh)
    k = max(1, int(len(df) * args.top_k_ratio))
    worst_df = df.head(k).copy()

    df.to_parquet(args.full_output_path)
    worst_df.to_parquet(args.worst_output_path)
    print(f"Saved full dataset to {args.full_output_path}")
    print(f"Saved top {args.top_k_ratio*100}% worst dataset to {args.worst_output_path}")

if __name__ == "__main__":
    main()
