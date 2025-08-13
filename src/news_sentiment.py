# src/news_sentiment.py
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Optional: NewsAPI (or your own news source)
import requests

# finBERT (graceful fallback)
_FINBERT_READY = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _clf = pipeline("text-classification", model=_model, tokenizer=_tokenizer, return_all_scores=True)
    _FINBERT_READY = True
except Exception:
    _FINBERT_READY = False

def _simple_polarity_heuristic(text: str) -> float:
    """Very small fallback sentiment scorer in [-1, 1]."""
    if not text:
        return 0.0
    text = text.lower()
    pos = sum(text.count(w) for w in ["beat", "surge", "record", "increase", "growth", "strong", "bull"])
    neg = sum(text.count(w) for w in ["miss", "fall", "drop", "decline", "weak", "bear", "cut", "lawsuit"])
    score = (pos - neg)
    if score == 0:
        return 0.0
    return float(np.tanh(score / 5.0))

def _finbert_score(texts: list[str]) -> list[float]:
    """Return sentiment scores in [-1, 1] using finBERT, else heuristic."""
    if _FINBERT_READY:
        scores = []
        for chunk in (texts[i:i+8] for i in range(0, len(texts), 8)):
            preds = _clf(chunk, truncation=True)
            for p in preds:
                # finBERT labels: [negative, neutral, positive]
                lab2score = {d["label"].lower(): d["score"] for d in p}
                score = lab2score.get("positive", 0) - lab2score.get("negative", 0)
                scores.append(float(score))
        return scores
    # fallback
    return [_simple_polarity_heuristic(t) for t in texts]

def fetch_recent_news(ticker: str, api_key: str | None, days: int = 7, limit: int = 25) -> pd.DataFrame:
    """Fetch recent news via NewsAPI (or return empty if no key)."""
    if not api_key:
        return pd.DataFrame(columns=["publishedAt", "title", "source"])
    q = f"{ticker} stock"
    frm = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = ("https://newsapi.org/v2/everything?"
           f"q={q}&from={frm}&sortBy=publishedAt&language=en&pageSize={limit}")
    headers = {"X-Api-Key": api_key}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
        rows = [{
            "publishedAt": a.get("publishedAt"),
            "title": a.get("title", ""),
            "source": (a.get("source") or {}).get("name", "")
        } for a in arts]
        df = pd.DataFrame(rows)
        if "publishedAt" in df.columns:
            df["publishedAt"] = pd.to_datetime(df["publishedAt"])
            df = df.sort_values("publishedAt")
        return df
    except Exception:
        return pd.DataFrame(columns=["publishedAt", "title", "source"])

def get_sentiment_score(ticker: str, api_key: str | None = None, days: int = 7) -> float:
    """Average sentiment over recent news headlines for ticker in [-1,1]."""
    df = fetch_recent_news(ticker, api_key, days=days)
    if df.empty:
        return 0.0
    titles = df["title"].fillna("").tolist()
    scores = _finbert_score(titles)
    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), -1.0, 1.0))
