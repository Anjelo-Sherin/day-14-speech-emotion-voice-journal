import pytest
from src.sentiment import analyze_text_sentiment

def test_positive_sentiment():
    text = "I am so happy and excited about this wonderful accomplishment!"
    res = analyze_text_sentiment(text)
    assert res["sentiment"] == "positive"
    assert res["score"] > 0.3
    assert res["text_emotion"] in ["joy", "positive"]

def test_negative_sentiment_and_negation():
    text = "I am deeply sad and exhausted from this stressful week."
    res = analyze_text_sentiment(text)
    assert res["sentiment"] == "negative"
    assert res["score"] < -0.2

    # Negation
    neg_text = "I am not happy with how things turned out."
    neg_res = analyze_text_sentiment(neg_text)
    assert neg_res["sentiment"] == "negative"

def test_masking_phrases():
    text = "I'm totally fine, don't worry about me at all."
    res = analyze_text_sentiment(text)
    assert res["has_masking_phrases"] is True
