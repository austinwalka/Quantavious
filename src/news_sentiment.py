# src/news_sentiment.py
import requests
import numpy as np

# You can sign up for free API keys on NewsAPI.org or similar services.
# Here is a placeholder using NewsAPI.org (replace YOUR_API_KEY)
NEWS_API_KEY = "7edaec35d5be4ee8846cbec2eed677d1"

def get_sentiment_score(ticker):
    """
    Fetch recent news headlines for ticker and compute a simple sentiment score.
    Returns a float in [-1,1]. Neutral is 0.
    """
    if NEWS_API_KEY == "YOUR_NEWSAPI_KEY":
        # Placeholder fallback - no API key set
        return 0.0

    url = ("https://newsapi.org/v2/everything?"
           f"q={ticker}&"
           "sortBy=publishedAt&"
           "language=en&"
           "pageSize=20&"
           f"apiKey={NEWS_API_KEY}")
    try:
        response = requests.get(url)
        data = response.json()
        articles = data.get('articles', [])
        if not articles:
            return 0.0

        # Simple sentiment using word lists (for demo)
        positive_words = {'good', 'great', 'positive', 'gain', 'up', 'beat', 'bull', 'buy'}
        negative_words = {'bad', 'poor', 'negative', 'loss', 'down', 'miss', 'bear', 'sell'}
        scores = []
        for art in articles:
            title = art.get('title', '').lower()
            body = art.get('description', '').lower()
            text = title + " " + body
            pos = sum(word in text for word in positive_words)
            neg = sum(word in text for word in negative_words)
            score = (pos - neg) / max(1, pos + neg)
            scores.append(score)
        # Average and clip
        avg_score = np.mean(scores)
        avg_score = max(min(avg_score, 1.0), -1.0)
        return avg_score
    except Exception:
        return 0.0
