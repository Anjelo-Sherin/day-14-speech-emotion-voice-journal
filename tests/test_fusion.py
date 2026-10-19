import pytest
from src.multimodal_fusion import fuse_audio_and_text

def test_masked_distress_detection():
    # User says cheerful words, but voice tone is sad and strained
    audio = {
        "dominant_emotion": "sad",
        "audio_valence": -0.65,
        "audio_arousal": 0.25,
        "confidence": 0.85,
        "emotion_breakdown": {"sad": 0.85, "happy": 0.05}
    }
    text = {
        "sentiment": "positive",
        "score": 0.55,
        "text_emotion": "joy",
        "has_masking_phrases": True
    }

    fusion = fuse_audio_and_text(audio, text)
    assert fusion["is_mismatch"] is True
    assert fusion["mismatch_type"] == "masked_distress"
    assert fusion["congruence_score"] < 65.0
    assert "suppressed" in fusion["fused_emotion"] or "distress" in fusion["fused_emotion"]

def test_congruent_joy():
    # User says happy words, voice sounds happy
    audio = {
        "dominant_emotion": "happy",
        "audio_valence": 0.75,
        "audio_arousal": 0.80,
        "confidence": 0.90,
        "emotion_breakdown": {"happy": 0.90}
    }
    text = {
        "sentiment": "positive",
        "score": 0.80,
        "text_emotion": "joy",
        "has_masking_phrases": False
    }

    fusion = fuse_audio_and_text(audio, text)
    assert fusion["is_mismatch"] is False
    assert fusion["mismatch_type"] == "congruent_positive"
    assert fusion["congruence_score"] > 80.0
    assert fusion["fused_valence"] > 0.5
