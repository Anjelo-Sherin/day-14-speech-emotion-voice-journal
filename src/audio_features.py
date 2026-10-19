import math
import struct
import wave
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.fft import dct

from src.config import settings

def load_audio(file_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a 1D float32 numpy array normalized to [-1.0, 1.0].
    Supports WAV natively; handles multi-channel downmixing and resampling.
    """
    file_path = Path(file_path)
    
    try:
        sr, data = wavfile.read(str(file_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
            max_val = np.max(np.abs(data))
            if max_val > 1.0:
                data = data / max_val
                
        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples).astype(np.float32)
            sr = target_sr
            
        return data, sr
    except Exception as e:
        try:
            with wave.open(str(file_path), 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                fmt = f"<{n_frames * n_channels}h"
                samples = np.array(struct.unpack(fmt, raw_bytes), dtype=np.float32) / 32768.0
            elif sampwidth == 1:
                fmt = f"<{n_frames * n_channels}B"
                samples = (np.array(struct.unpack(fmt, raw_bytes), dtype=np.float32) - 128.0) / 128.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            if sr != target_sr and len(samples) > 0:
                num_samples = int(len(samples) * target_sr / sr)
                samples = signal.resample(samples, num_samples).astype(np.float32)
                sr = target_sr

            return samples, sr
        except Exception as inner_e:
            raise RuntimeError(f"Could not read audio file {file_path}: {e}; fallback error: {inner_e}")

def create_mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    """
    Construct triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    num_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        if center > left:
            filterbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filterbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]
    return filterbank

def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64
) -> np.ndarray:
    """
    Compute log-mel power spectrogram (shape: n_mels, time_steps).
    """
    if len(y) == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)
    
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    f, t, Zxx = signal.stft(
        y_pre,
        fs=sr,
        window='hann',
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=True
    )
    power_spec = np.abs(Zxx) ** 2

    fb = create_mel_filterbank(sr, n_fft, n_mels=n_mels)
    mel_spec = np.dot(fb, power_spec)
    mel_spec = np.maximum(mel_spec, 1e-10)

    log_mel_spec = 10.0 * np.log10(mel_spec)
    log_mel_spec = np.maximum(log_mel_spec, log_mel_spec.max() - 80.0)
    return log_mel_spec.astype(np.float32)

def compute_mfcc(
    y: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute Mel-Frequency Cepstral Coefficients (MFCCs). Shape: (n_mfcc, time_steps)
    """
    log_mel_spec = compute_mel_spectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mfcc_coeffs = dct(log_mel_spec, type=2, axis=0, norm='ortho')[:n_mfcc]
    return mfcc_coeffs.astype(np.float32)

def compute_acoustic_prosody(y: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Compute prosodic and voice tone descriptors:
    - RMS Energy (loudness & dynamics)
    - Zero Crossing Rate (friction & articulation)
    - Spectral Centroid (spectral brightness)
    - Estimated Pitch F0 stats (fundamental frequency, variability, arousal)
    """
    if len(y) == 0:
        return {
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "zcr_mean": 0.0,
            "spectral_centroid_mean": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "estimated_arousal": 0.5,
            "estimated_valence": 0.0
        }

    # 1. RMS Energy
    frame_size = 1024
    hop_size = 512
    n_frames = max(1, (len(y) - frame_size) // hop_size + 1)
    
    rms = np.array([
        np.sqrt(np.mean(y[i * hop_size : i * hop_size + frame_size] ** 2) + 1e-12)
        for i in range(n_frames)
    ])
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # 2. Zero Crossing Rate (ZCR)
    zero_crossings = np.abs(np.diff(np.signbit(y)))
    zcr_mean = float(np.mean(zero_crossings))

    # 3. Spectral Centroid (speech-band limited 100 Hz - 4000 Hz)
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size)
    magnitude = np.abs(Zxx)
    freq_bins = f[:, np.newaxis]
    sum_mag = np.sum(magnitude, axis=0) + 1e-10
    spectral_centroid = np.sum(freq_bins * magnitude, axis=0) / sum_mag
    spec_cent_mean = float(np.mean(spectral_centroid))

    # 4. Pitch (F0) estimation via autocorrelation in 60-400 Hz range
    min_lag = int(sr / 400.0)  # 400 Hz
    max_lag = int(sr / 60.0)   # 60 Hz
    pitches = []
    
    for i in range(0, len(y) - frame_size, hop_size * 2):
        chunk = y[i : i + frame_size]
        if np.max(np.abs(chunk)) < 0.01:
            continue
        corr = signal.correlate(chunk, chunk, mode='full')[frame_size - 1:]
        lag_window = corr[min_lag:max_lag]
        if len(lag_window) > 0:
            peak_lag = min_lag + np.argmax(lag_window)
            f0 = sr / peak_lag
            if 60 <= f0 <= 380:
                pitches.append(f0)

    if pitches:
        pitch_mean = float(np.mean(pitches))
        pitch_std = float(np.std(pitches))
    else:
        pitch_mean = 140.0
        pitch_std = 15.0

    # Calibrated Acoustic Arousal:
    # High pitch + high energy + high pitch std -> high arousal (excitement, anger, panic)
    # Low pitch + low energy + flat pitch std -> low arousal (sadness, calm, fatigue)
    norm_pitch = np.clip((pitch_mean - 100.0) / 140.0, 0.0, 1.0)
    norm_rms = np.clip(rms_mean * 12.0, 0.0, 1.0)
    norm_pitch_std = np.clip(pitch_std / 35.0, 0.0, 1.0)

    estimated_arousal = float(np.clip(0.35 * norm_rms + 0.35 * norm_pitch + 0.30 * norm_pitch_std, 0.05, 0.98))
    
    # Calibrated Acoustic Valence:
    # High pitch variability + open dynamics = positive valence (joy, pleasant)
    # Monotonic, flat pitch (<10 Hz) + low energy = negative valence (sad, fatigued)
    # Very high pitch + harsh centroid + high energy = negative valence (angry, stressed)
    if pitch_std < 10.0 and pitch_mean < 140.0:
        # Flat, depressed/sad vocal tone
        valence_pitch = -0.6
    elif norm_pitch > 0.7 and norm_pitch_std > 0.6 and norm_rms > 0.6:
        valence_pitch = 0.5  # animated joy
    else:
        valence_pitch = (norm_pitch_std - 0.3) * 1.2

    estimated_valence = float(np.clip(valence_pitch, -1.0, 1.0))

    return {
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "zcr_mean": round(zcr_mean, 5),
        "spectral_centroid_mean": round(spec_cent_mean, 2),
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "estimated_arousal": round(estimated_arousal, 3),
        "estimated_valence": round(estimated_valence, 3)
    }

def extract_features_for_model(
    y: np.ndarray,
    sr: int = 16000,
    target_frames: int = 128,
    n_mfcc: int = 40
) -> np.ndarray:
    """
    Extract fixed-length (n_mfcc, target_frames) matrix for CNN-LSTM model.
    Pads or truncates time dimension to target_frames.
    Returns array of shape (1, n_mfcc, target_frames) ready for PyTorch.
    """
    mfccs = compute_mfcc(y, sr=sr, n_mfcc=n_mfcc, n_mels=64)
    curr_frames = mfccs.shape[1]
    if curr_frames < target_frames:
        pad_width = target_frames - curr_frames
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :target_frames]

    mean = np.mean(mfccs, axis=1, keepdims=True)
    std = np.std(mfccs, axis=1, keepdims=True) + 1e-6
    mfccs = (mfccs - mean) / std

    return np.expand_dims(mfccs, axis=0).astype(np.float32)
