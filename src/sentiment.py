import re
from typing import Dict, Any, List, Tuple

# Comprehensive sentiment and emotion lexicon tailored for voice journaling
EMOTION_LEXICON = {
    # Joy / Positive
    "happy": ("joy", 0.8), "great": ("joy", 0.8), "excited": ("joy", 0.9), "wonderful": ("joy", 0.9),
    "good": ("joy", 0.5), "grateful": ("joy", 0.75), "blessed": ("joy", 0.8), "fantastic": ("joy", 0.95),
    "amazing": ("joy", 0.9), "proud": ("joy", 0.7), "love": ("joy", 0.85), "peaceful": ("calm", 0.7),
    "calm": ("calm", 0.65), "relaxed": ("calm", 0.7), "fine": ("neutral", 0.2), "okay": ("neutral", 0.1),
    "alright": ("neutral", 0.15), "content": ("calm", 0.6), "hopeful": ("joy", 0.7),

    # Sadness / Depressive
    "sad": ("sadness", -0.8), "depressed": ("sadness", -0.9), "unhappy": ("sadness", -0.7),
    "crying": ("sadness", -0.85), "tears": ("sadness", -0.75), "lonely": ("sadness", -0.8),
    "hopeless": ("sadness", -0.95), "heartbroken": ("sadness", -0.9), "grief": ("sadness", -0.9),
    "exhausted": ("sadness", -0.65), "drained": ("sadness", -0.7), "miserable": ("sadness", -0.85),
    "empty": ("sadness", -0.7), "tired": ("sadness", -0.4), "hurting": ("sadness", -0.8),

    # Anger / Frustration
    "angry": ("anger", -0.85), "furious": ("anger", -0.95), "mad": ("anger", -0.75),
    "frustrated": ("anger", -0.7), "annoyed": ("anger", -0.6), "irritated": ("anger", -0.65),
    "hate": ("anger", -0.9), "rage": ("anger", -0.95), "disgusted": ("disgust", -0.8),
    "terrible": ("anger", -0.75), "awful": ("anger", -0.8), "horrible": ("anger", -0.85),

    # Fear / Anxiety / Stress
    "anxious": ("fear", -0.75), "nervous": ("fear", -0.6), "worried": ("fear", -0.65),
    "scared": ("fear", -0.8), "terrified": ("fear", -0.9), "panic": ("fear", -0.9),
    "stressed": ("fear", -0.7), "overwhelmed": ("fear", -0.8), "pressure": ("fear", -0.6),
    "afraid": ("fear", -0.8), "jittery": ("fear", -0.65), "uneasy": ("fear", -0.55),

    # Surprise
    "shocked": ("surprise", -0.2), "surprised": ("surprise", 0.3), "unexpected": ("surprise", 0.1),
    "unbelievable": ("surprise", 0.4), "astonished": ("surprise", 0.5)
}

NEGATIONS = {
    "not", "no", "never", "don't", "dont", "didn't", "didnt", "can't", "cant",
    "won't", "wont", "hardly", "barely", "scarcely", "nothing", "nowhere"
}

INTENSIFIERS = {
    "very": 1.4, "really": 1.3, "extremely": 1.6, "so": 1.3, "totally": 1.4,
    "super": 1.4, "absolutely": 1.5, "deeply": 1.5, "quite": 1.2
}

# Common masking phrases where words appear cheerful or nonchalant
MASKING_PHRASES = [
    "i'm fine", "im fine", "everything is fine", "everything's fine",
    "i'm okay", "im okay", "don't worry about me", "no problem at all",
    "totally fine", "it's nothing", "just tired"
]

def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Evaluates semantic sentiment, emotion category, and valence for text.
    Handles negations, intensifiers, and identifies emotional keywords.
    """
    clean_text = text.strip()
    if not clean_text:
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "text_emotion": "neutral",
            "detected_keywords": [],
            "word_count": 0,
            "has_masking_phrases": False
        }

    words = re.findall(r"\b[\w'-]+\b", clean_text.lower())
    valence_scores = []
    emotion_counts: Dict[str, float] = {
        "joy": 0.0, "calm": 0.0, "sadness": 0.0,
        "anger": 0.0, "fear": 0.0, "surprise": 0.0, "neutral": 0.05
    }
    detected_keywords = []

    # Check for masking phrase matches
    text_lower = clean_text.lower()
    has_masking = any(phrase in text_lower for phrase in MASKING_PHRASES)

    i = 0
    while i < len(words):
        word = words[i]
        multiplier = 1.0
        is_negated = False

        # Look back 1-2 words for negation and intensifier
        lookback = words[max(0, i - 2) : i]
        for prev in lookback:
            if prev in NEGATIONS:
                is_negated = True
            if prev in INTENSIFIERS:
                multiplier *= INTENSIFIERS[prev]

        if word in EMOTION_LEXICON:
            emo, base_val = EMOTION_LEXICON[word]
            score = base_val * multiplier
            if is_negated:
                score = -score * 0.8
                # Invert emotion
                if emo in ["joy", "calm"]:
                    emo = "sadness"
                elif emo in ["sadness", "anger", "fear"]:
                    emo = "calm"

            valence_scores.append(score)
            emotion_counts[emo] = emotion_counts.get(emo, 0.0) + abs(score)
            detected_keywords.append({
                "word": word,
                "emotion": emo,
                "score": round(score, 2),
                "negated": is_negated
            })
        i += 1

    if valence_scores:
        avg_score = float(np_mean(valence_scores))
    else:
        avg_score = 0.0

    # Classify overall sentiment
    if avg_score > 0.15:
        sentiment = "positive"
    elif avg_score < -0.15:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Dominant text emotion
    dominant_emo = max(emotion_counts.items(), key=lambda item: item[1])[0]
    if dominant_emo == "joy" and avg_score < 0:
        dominant_emo = "sadness"

    return {
        "sentiment": sentiment,
        "score": round(float(avg_score), 3),
        "text_emotion": dominant_emo,
        "detected_keywords": detected_keywords,
        "word_count": len(words),
        "has_masking_phrases": has_masking
    }

def np_mean(vals: List[float]) -> float:
    return sum(vals) / max(1, len(vals))
