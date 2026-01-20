from langchain.tools import tool
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import get_stock_data  # Import from task1

@tool
def fetch_news(ticker):
    """
    Fetches news for a ticker and analyzes sentiment.
    Requires NewsAPI key.
    """
    api_key = "YOUR_NEWSAPI_KEY"  # Replace with your key
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url).json()
    articles = response.get('articles', [])
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    for art in articles[:5]:
        sent = analyzer.polarity_scores(art['description'] or "")['compound']
        news_data.append({"title": art['title'], "desc": art['description'], "sent": sent})
    return news_data

@tool
def analyze_stock(ticker, query):
    """
    Analyzes stock data for trends based on query.
    """
    data = get_stock_data(ticker)
    if "drop" in query.lower():
        drops = data[data['Close'].pct_change() < -0.05]
        return f"Drops on: {drops.index.tolist()}"
    elif "up" in query.lower():
        ups = data[data['Close'].pct_change() > 0.05]
        return f"Ups on: {ups.index.tolist()}"
    return "No specific trend found."
