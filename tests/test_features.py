import pytest
import numpy as np
from pathlib import Path

from src.audio_features import (
    create_mel_filterbank,
    compute_mel_spectrogram,
    compute_mfcc,
    compute_acoustic_prosody,
    extract_features_for_model
)

def test_mel_filterbank():
    sr = 16000
    n_fft = 1024
    n_mels = 64
    fb = create_mel_filterbank(sr, n_fft, n_mels)
    assert fb.shape == (n_mels, n_fft // 2 + 1)
    assert not np.isnan(fb).any()

def test_spectrogram_and_mfcc():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # 200 Hz tone
    y = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    mel_spec = compute_mel_spectrogram(y, sr=sr, n_mels=64)
    assert mel_spec.shape[0] == 64
    assert mel_spec.shape[1] > 10

    mfccs = compute_mfcc(y, sr=sr, n_mfcc=40, n_mels=64)
    assert mfccs.shape[0] == 40
    assert mfccs.shape[1] > 10

def test_acoustic_prosody():
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration))
    # Modulated tone
    y = (0.4 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.random.normal(0, 0.05, len(t))).astype(np.float32)

    prosody = compute_acoustic_prosody(y, sr)
    assert "rms_mean" in prosody
    assert "pitch_mean_hz" in prosody
    assert "estimated_arousal" in prosody
    assert "estimated_valence" in prosody
    assert 0.0 <= prosody["estimated_arousal"] <= 1.0
    assert -1.0 <= prosody["estimated_valence"] <= 1.0

def test_feature_tensor_extraction():
    sr = 16000
    y = np.random.uniform(-0.5, 0.5, sr * 2).astype(np.float32)
    feats = extract_features_for_model(y, sr=sr, target_frames=128, n_mfcc=40)
    assert feats.shape == (1, 40, 128)
    assert feats.dtype == np.float32
