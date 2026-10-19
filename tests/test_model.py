import pytest
import torch
import numpy as np

from src.model import SpeechEmotionCNNLSTM, SpeechEmotionPredictor
from src.config import settings

def test_cnn_lstm_forward():
    model = SpeechEmotionCNNLSTM(num_classes=8)
    model.eval()

    # Input batch of 2 samples, 1 channel, 40 MFCCs, 128 frames
    x = torch.randn(2, 1, 40, 128)
    with torch.no_grad():
        logits, attn = model(x)

    assert logits.shape == (2, 8)
    assert attn.shape == (2, 16)
    assert torch.allclose(torch.sum(attn, dim=1), torch.ones(2), atol=1e-4)

def test_speech_emotion_predictor():
    predictor = SpeechEmotionPredictor()
    dummy_feats = np.random.randn(1, 40, 128).astype(np.float32)
    dummy_prosody = {
        "estimated_arousal": 0.7,
        "estimated_valence": 0.4,
        "pitch_mean_hz": 220.0
    }

    result = predictor.predict(dummy_feats, dummy_prosody)
    assert "dominant_emotion" in result
    assert result["dominant_emotion"] in settings.EMOTIONS
    assert 0.0 <= result["confidence"] <= 1.0
    assert "audio_arousal" in result
    assert "audio_valence" in result
    assert len(result["emotion_breakdown"]) == len(settings.EMOTIONS)
