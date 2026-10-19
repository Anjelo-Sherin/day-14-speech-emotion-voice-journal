import os
import json
from typing import Dict, Any, Optional
import httpx

from src.config import settings

def generate_reflective_summary(
    transcript: str,
    audio_emotion: str,
    audio_confidence: float,
    text_sentiment: str,
    fused_emotion: str,
    congruence_score: float,
    is_mismatch: bool,
    mismatch_type: str
) -> str:
    """
    Generates an empathetic, personalized 2-3 sentence reflective journal entry summary.
    Uses Google Gemini API if GEMINI_API_KEY is configured, else uses an intelligent
    deterministic reflective wellness synthesis engine.
    """
    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

    if api_key and len(api_key.strip()) > 10:
        try:
            summary = _call_gemini_api(
                api_key=api_key.strip(),
                transcript=transcript,
                audio_emotion=audio_emotion,
                text_sentiment=text_sentiment,
                fused_emotion=fused_emotion,
                congruence_score=congruence_score,
                is_mismatch=is_mismatch,
                mismatch_type=mismatch_type
            )
            if summary and len(summary) > 20:
                return summary
        except Exception as e:
            print(f"[ReflectiveSummary] Gemini API call failed: {e}. Falling back to internal engine.")

    # High-quality empathetic local engine
    return _generate_local_reflective_summary(
        transcript=transcript,
        audio_emotion=audio_emotion,
        text_sentiment=text_sentiment,
        fused_emotion=fused_emotion,
        congruence_score=congruence_score,
        is_mismatch=is_mismatch,
        mismatch_type=mismatch_type
    )

def _generate_local_reflective_summary(
    transcript: str,
    audio_emotion: str,
    text_sentiment: str,
    fused_emotion: str,
    congruence_score: float,
    is_mismatch: bool,
    mismatch_type: str
) -> str:
    """
    Synthesizes an empathetic, grounded reflection without external API dependencies.
    """
    if is_mismatch and mismatch_type == "masked_distress":
        return (
            f"You spoke reassuring words about your day, but your vocal acoustics carried unmistakable "
            f"traces of {audio_emotion} and fatigue ({congruence_score:.0f}% congruence). "
            f"It takes energy to hold things together on the surface; honor what your body is carrying "
            f"and grant yourself permission to step back and unwind this evening."
        )

    if is_mismatch and mismatch_type == "suppressed_frustration":
        return (
            f"While your words remained diplomatic and composed, your vocal dynamics revealed underlying "
            f"agitation and sharp energy. Recognizing subtle frustration early prevents emotional exhaustion. "
            f"Consider taking a few long exhales before diving into the next project."
        )

    if is_mismatch and mismatch_type == "somber_acceptance":
        return (
            f"Even while recounting a challenging or somber experience, your vocal tone was remarkably "
            f"peaceful and steady. This calm resonance indicates genuine emotional maturity and "
            f"resilience in navigating difficulty."
        )

    if fused_emotion in ["authentic_joy", "happy"]:
        return (
            f"Your voice echoed genuine warmth and positive vibrancy today, perfectly aligned with your words! "
            f"Moments of authentic clarity and joy nourish your cognitive stamina—take a mental snapshot of "
            f"what made today feel rewarding."
        )

    if fused_emotion in ["calm_contentment", "calm"]:
        return (
            f"A deeply grounded, serene journal entry. Both your vocal cadence and message reflect a balanced, "
            f"restorative mindset. Maintain this gentle rhythm as you move through your week."
        )

    if fused_emotion in ["authentic_sad", "sad"]:
        return (
            f"You spoke with authentic vulnerability today. Acknowledging sadness through your voice and words "
            f"is a courageous form of self-honesty that initiates healing. Be especially kind and gentle with yourself today."
        )

    if fused_emotion in ["authentic_angry", "angry"]:
        return (
            f"Your entry carried palpable intensity and drive. Vocal expression is a healthy outlet for release; "
            f"use this kinetic energy constructively while giving your nervous system time to settle."
        )

    if fused_emotion in ["authentic_fearful", "fearful"]:
        return (
            f"Acoustic cues reflected elevated nervousness and apprehension. When feelings of uncertainty rise, "
            f"focus exclusively on the next immediate small step you can control."
        )

    # General fallback
    return (
        f"Today's voice journal entry captured a steady, reflective cadence ({fused_emotion.replace('_', ' ')}). "
        f"Consistently tracking both your spoken words and tonal resonance builds lasting emotional self-awareness."
    )

def _call_gemini_api(
    api_key: str,
    transcript: str,
    audio_emotion: str,
    text_sentiment: str,
    fused_emotion: str,
    congruence_score: float,
    is_mismatch: bool,
    mismatch_type: str
) -> Optional[str]:
    """
    Calls Gemini API via REST endpoint with structured emotional metadata.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    prompt = (
        f"You are an empathetic, clinically-safe, mindful voice journaling companion (wellness aid, not a medical doctor).\n"
        f"The user recorded a personal voice note with the following multimodal analysis:\n"
        f"- Spoken Transcript: \"{transcript}\"\n"
        f"- Vocal Tone Emotion (from acoustics): {audio_emotion}\n"
        f"- Lexical Sentiment (from words): {text_sentiment}\n"
        f"- Multimodal Fused Emotion: {fused_emotion}\n"
        f"- Affective Congruence Score: {congruence_score}%\n"
        f"- Detected Mismatch: {mismatch_type if is_mismatch else 'None (Congruent)'}\n\n"
        f"Write a warm, thoughtful 2-to-3 sentence reflective summary for their voice journal entry. "
        f"If there is a mismatch (e.g. positive words with stressed/sad tone), gently notice it without judgment. "
        f"Provide one small, comforting self-care insight. Keep it concise, natural, and supportive."
    )

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 200
        }
    }

    headers = {"Content-Type": "application/json"}
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return None
