from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
import numpy as np
def fetch_sentiment_score(query):
    googlenews = GoogleNews(period='7d')
    googlenews.search(query)
    articles = googlenews.result()
    analyzer = SentimentIntensityAnalyzer()
    
    scores = []
    for article in articles:
        text = article['title']
        vs = analyzer.polarity_scores(text)
        scores.append(vs['compound'])
    return round(np.mean(scores), 2) if scores else 0.0