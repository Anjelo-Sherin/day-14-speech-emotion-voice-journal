import math
import struct
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from src.config import settings
from src.database import SessionLocal, JournalEntry, init_db
from src.audio_features import load_audio, compute_acoustic_prosody, extract_features_for_model
from src.model import predictor
from src.sentiment import analyze_text_sentiment
from src.multimodal_fusion import fuse_audio_and_text
from src.reflective_summary import generate_reflective_summary

def create_synthetic_wav(
    filepath: Path,
    duration: float = 3.5,
    sample_rate: int = 16000,
    base_freq: float = 180.0,
    pitch_mod: float = 15.0,
    mod_speed: float = 3.0,
    amplitude: float = 0.5,
    tremolo_speed: float = 5.0,
    noise_level: float = 0.02
) -> Path:
    """
    Synthesize speech-like formants and prosodic frequency modulation into a clean WAV file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Prosodic pitch contour: base frequency + slow inflection
    f_contour = base_freq + pitch_mod * np.sin(2 * np.pi * mod_speed * t)
    phase = 2 * np.pi * np.cumsum(f_contour) / sample_rate

    # Fundamental voice formant harmonics (F0, F1 ~ 3*F0, F2 ~ 5*F0)
    signal = (
        0.55 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.15 * np.sin(3 * phase) +
        0.05 * np.sin(4 * phase)
    )

    # Vocal tract tremolo / amplitude envelope resembling syllables
    syllables = (np.sin(2 * np.pi * tremolo_speed * t) + 1.0) / 2.0
    syllables = np.clip(syllables ** 1.5, 0.1, 1.0)
    signal = signal * syllables * amplitude

    # Add gentle pink/white breathing noise
    noise = np.random.normal(0, noise_level, n_samples)
    signal = signal + noise

    # Fade in / fade out to avoid clicks
    fade_len = int(0.05 * sample_rate)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out

    # Normalize to 16-bit PCM
    signal = np.clip(signal, -0.98, 0.98)
    int16_data = (signal * 32767).astype(np.int16)

    with wave.open(str(filepath), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    return filepath

SAMPLE_PRESETS = [
    {
        "id": "masked_distress",
        "title": "Masked Distress ('I'm totally fine')",
        "transcript": "Everything is totally fine. I am handling work and no problems at all with the deadline, really I'm doing great.",
        "params": {
            "base_freq": 115.0,  # low, strained pitch
            "pitch_mod": 6.0,    # constricted pitch range
            "mod_speed": 1.5,
            "amplitude": 0.28,   # subdued, low energy
            "tremolo_speed": 3.0,
            "noise_level": 0.04
        }
    },
    {
        "id": "authentic_joy",
        "title": "Authentic Joy ('We got funded!')",
        "transcript": "I am so excited and happy! We finally crossed our milestones and celebrated with the entire team today. What an incredible feeling!",
        "params": {
            "base_freq": 240.0,  # bright, high pitch
            "pitch_mod": 45.0,   # animated pitch variance
            "mod_speed": 4.5,
            "amplitude": 0.75,   # rich, full acoustic energy
            "tremolo_speed": 5.5,
            "noise_level": 0.015
        }
    },
    {
        "id": "mindful_calm",
        "title": "Mindful Equilibrium ('Peaceful morning')",
        "transcript": "Taking a slow walk through the park this morning. Breathing in the fresh cool air and feeling calm, content, and grounded.",
        "params": {
            "base_freq": 140.0,  # balanced fundamental
            "pitch_mod": 12.0,   # smooth melodic contour
            "mod_speed": 1.8,
            "amplitude": 0.40,   # moderate, soothing dynamics
            "tremolo_speed": 2.2,
            "noise_level": 0.01
        }
    },
    {
        "id": "suppressed_frustration",
        "title": "Suppressed Frustration ('Handling the delay')",
        "transcript": "I reviewed the requested changes again. It is manageable and I understand their position, but the delays are becoming very noticeable.",
        "params": {
            "base_freq": 210.0,  # elevated, tense pitch
            "pitch_mod": 35.0,   # sharp bursts
            "mod_speed": 5.0,
            "amplitude": 0.65,   # forceful energy
            "tremolo_speed": 6.0,
            "noise_level": 0.03
        }
    }
]

def generate_preset_samples():
    """
    Pre-synthesizes the 4 demonstration WAV samples into data/audio_uploads.
    """
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for preset in SAMPLE_PRESETS:
        wav_path = settings.UPLOAD_DIR / f"sample_{preset['id']}.wav"
        if not wav_path.exists():
            create_synthetic_wav(
                filepath=wav_path,
                duration=4.0,
                sample_rate=settings.SAMPLE_RATE,
                **preset["params"]
            )
        generated.append({
            "id": preset["id"],
            "title": preset["title"],
            "transcript": preset["transcript"],
            "audio_path": f"/static/audio/{wav_path.name}"
        })
    return generated

def seed_database_if_empty():
    """
    Populates SQLite database with a realistic 14-day history if empty,
    demonstrating the weekly mood trend and calendar heatmap out-of-the-box.
    """
    init_db()
    db = SessionLocal()
    try:
        count = db.query(JournalEntry).count()
        if count > 0:
            return

        print("[Seed] Seeding realistic 14-day voice journal history...")
        generate_preset_samples()

        now = datetime.utcnow()
        historical_records = [
            ("Today (Morning)", 0, "I am so excited and happy! We crossed our major milestones today.", "sample_authentic_joy.wav"),
            ("Yesterday", 1, "Everything is totally fine. Just wrapping up some tasks.", "sample_masked_distress.wav"),
            ("2 Days Ago", 2, "Taking a quiet mindful stroll. Feeling very peaceful and refreshed.", "sample_mindful_calm.wav"),
            ("3 Days Ago", 3, "Reviewed the delay in deliverables. Handling it step by step.", "sample_suppressed_frustration.wav"),
            ("4 Days Ago", 4, "Woke up early and finished my workout. Steady productive day.", "sample_mindful_calm.wav"),
            ("5 Days Ago", 5, "Felt a bit tired in the afternoon but pushed through.", "sample_masked_distress.wav"),
            ("6 Days Ago", 6, "Met an old college friend for coffee! Really wonderful conversation.", "sample_authentic_joy.wav"),
            ("7 Days Ago", 7, "Sunday evening planning for the week ahead. Calm and centered.", "sample_mindful_calm.wav"),
            ("8 Days Ago", 8, "Busy sprint planning. High energy in the room.", "sample_authentic_joy.wav"),
            ("9 Days Ago", 9, "Stressed about the client deadline but sticking to the plan.", "sample_suppressed_frustration.wav"),
            ("10 Days Ago", 10, "Feeling good after meditating and journaling.", "sample_mindful_calm.wav"),
            ("11 Days Ago", 11, "Just another routine day, keeping steady.", "sample_mindful_calm.wav"),
            ("12 Days Ago", 12, "Celebrated our team demo! Great feedback.", "sample_authentic_joy.wav"),
            ("13 Days Ago", 13, "Late night debugging, exhausted but satisfied.", "sample_masked_distress.wav"),
        ]

        for label, days_ago, text, wav_file in historical_records:
            entry_time = now - timedelta(days=days_ago, hours=np.random.randint(2, 6))
            wav_path = settings.UPLOAD_DIR / wav_file
            
            # Extract features
            if wav_path.exists():
                y, sr = load_audio(wav_path, settings.SAMPLE_RATE)
                prosody = compute_acoustic_prosody(y, sr)
                feats = extract_features_for_model(y, sr, settings.TARGET_FRAMES, settings.N_MFCC)
                audio_res = predictor.predict(feats, prosody)
            else:
                prosody = {"estimated_arousal": 0.5, "estimated_valence": 0.0}
                audio_res = {
                    "dominant_emotion": "neutral",
                    "confidence": 0.65,
                    "emotion_breakdown": {"neutral": 0.65, "calm": 0.35},
                    "audio_arousal": 0.45,
                    "audio_valence": 0.0
                }

            text_res = analyze_text_sentiment(text)
            fusion_res = fuse_audio_and_text(audio_res, text_res)
            summary = generate_reflective_summary(
                transcript=text,
                audio_emotion=audio_res["dominant_emotion"],
                audio_confidence=audio_res["confidence"],
                text_sentiment=text_res["sentiment"],
                fused_emotion=fusion_res["fused_emotion"],
                congruence_score=fusion_res["congruence_score"],
                is_mismatch=fusion_res["is_mismatch"],
                mismatch_type=fusion_res["mismatch_type"]
            )

            entry = JournalEntry(
                created_at=entry_time,
                audio_filename=wav_file,
                audio_path=str(wav_path),
                duration_sec=3.8,
                transcript=text,
                audio_emotion=audio_res["dominant_emotion"],
                audio_confidence=audio_res["confidence"],
                audio_arousal=audio_res["audio_arousal"],
                audio_valence=audio_res["audio_valence"],
                text_sentiment=text_res["sentiment"],
                text_sentiment_score=text_res["score"],
                text_emotion=text_res["text_emotion"],
                fused_emotion=fusion_res["fused_emotion"],
                fused_valence=fusion_res["fused_valence"],
                fused_arousal=fusion_res["fused_arousal"],
                congruence_score=fusion_res["congruence_score"],
                is_mismatch=fusion_res["is_mismatch"],
                mismatch_type=fusion_res["mismatch_type"],
                mismatch_explanation=fusion_res["mismatch_explanation"],
                reflective_summary=summary,
                actionable_tip=fusion_res["actionable_tip"]
            )
            entry.emotion_breakdown = audio_res["emotion_breakdown"]
            db.add(entry)

        db.commit()
        print("[Seed] Successfully seeded 14 journal entries!")
    except Exception as e:
        db.rollback()
        print(f"[Seed] Error seeding database: {e}")
    finally:
        db.close()
