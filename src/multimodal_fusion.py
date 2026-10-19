import math
from typing import Dict, Any

def fuse_audio_and_text(
    audio_analysis: Dict[str, Any],
    text_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Multimodal fusion layer combining vocal acoustic cues (prosody + CNN-LSTM)
    with lexical text sentiment.
    Detects affective congruence, emotional masking, and cognitive-vocal mismatches.
    """
    audio_emo = audio_analysis.get("dominant_emotion", "neutral")
    audio_val = float(audio_analysis.get("audio_valence", 0.0))
    audio_aro = float(audio_analysis.get("audio_arousal", 0.5))
    audio_conf = float(audio_analysis.get("confidence", 0.5))
    breakdown = audio_analysis.get("emotion_breakdown", {})

    text_sent = text_analysis.get("sentiment", "neutral")
    text_val = float(text_analysis.get("score", 0.0))
    text_emo = text_analysis.get("text_emotion", "neutral")
    has_masking = text_analysis.get("has_masking_phrases", False)

    # 1. Compute Fused Valence
    w_audio = 0.60
    w_text = 0.40
    fused_valence = round(w_audio * audio_val + w_text * text_val, 3)

    # 2. Compute Fused Arousal
    text_intensity = abs(text_val)
    fused_arousal = round(0.75 * audio_aro + 0.25 * min(1.0, text_intensity + 0.3), 3)

    # 3. Calculate Congruence Score (0 - 100%)
    val_diff = abs(audio_val - text_val)
    congruence = max(10.0, 100.0 - (val_diff * 45.0))
    
    if text_val > 0.15 and audio_val < -0.05:
        congruence -= 25.0
    elif text_val < -0.15 and audio_val > 0.25:
        congruence -= 20.0

    if has_masking and audio_val <= 0.0:
        congruence -= 20.0

    congruence_score = round(max(5.0, min(100.0, congruence)), 1)

    # 4. Mismatch Classification
    is_mismatch = False
    mismatch_type = "congruent"
    explanation = ""
    fused_emotion = audio_emo

    # Case A: Masked Distress
    # Positive/neutral text, but vocal tone is sad/fearful/angry OR negative acoustic valence, OR masking phrases with low/negative valence
    if (text_val >= 0.1 and (audio_emo in ["sad", "fearful", "angry"] or audio_val <= -0.15)) or (has_masking and audio_val <= 0.05 and text_val > 0.15):
        is_mismatch = True
        mismatch_type = "masked_distress"
        fused_emotion = "suppressed_distress"
        explanation = (
            f"Vocal tone reveals underlying {audio_emo} strain (acoustic valence {audio_val:.2f}) "
            f"despite your positive words (sentiment {text_val:+.2f}). "
            f"This frequently signals emotional masking, fatigue, or unvoiced stress."
        )

    # Case B: Agitated / Suppressed Frustration (Polite words, high vocal arousal/pitch)
    elif text_val >= -0.1 and (audio_emo == "angry" or (audio_aro > 0.75 and audio_val < 0.0)):
        is_mismatch = True
        mismatch_type = "suppressed_frustration"
        fused_emotion = "frustration"
        explanation = (
            f"Your transcript maintained composed or neutral vocabulary, but your acoustic "
            f"inflections showed sharp vocal bursts and heightened arousal indicative of frustration."
        )

    # Case C: Somber Relief (Grave words, but serene/calm voice)
    elif text_val <= -0.3 and (audio_emo in ["calm", "neutral"] and audio_aro < 0.4):
        is_mismatch = True
        mismatch_type = "somber_acceptance"
        fused_emotion = "peaceful_acceptance"
        explanation = (
            f"While the words discuss difficult or challenging topics, your voice was steady, "
            f"grounded, and calm (low arousal {audio_aro:.2f}), suggesting emotional acceptance or peaceful closure."
        )

    # Case D: Sarcastic or Deadpan (Overjoyed words, flat/disgusted voice)
    elif text_val >= 0.5 and (audio_emo in ["disgust", "neutral"] and audio_aro < 0.4):
        is_mismatch = True
        mismatch_type = "sarcasm_or_detachment"
        fused_emotion = "ironic_detachment"
        explanation = (
            f"Hyper-positive vocabulary was delivered with a flat or detached vocal timbre, "
            f"frequently characteristic of irony, humor, or cynicism."
        )

    # Case E: Congruent Authentic Expressions
    else:
        is_mismatch = False
        if text_val > 0.2 and audio_val > 0.2:
            mismatch_type = "congruent_positive"
            fused_emotion = "authentic_joy" if audio_aro > 0.5 else "calm_contentment"
            explanation = "Your vocal inflection and written words are in high harmony, projecting genuine warmth and optimism."
        elif text_val < -0.2 and audio_val < -0.2:
            mismatch_type = "congruent_catharsis"
            fused_emotion = f"authentic_{audio_emo}"
            explanation = f"Your voice and words openly acknowledge feelings of {audio_emo}. Expressing this authentically aids emotional processing."
        else:
            mismatch_type = "congruent_balanced"
            fused_emotion = audio_emo
            explanation = "Voice prosody and lexical sentiment are balanced in an equilibrium state."

    # Generate tailored wellness micro-tip
    tips = {
        "masked_distress": "Take 3 minutes away from demands. Give yourself permission to feel tired or strained without having to 'fix' it immediately.",
        "suppressed_distress": "Write down the single biggest friction point right now on paper, then close the notebook. Unload the mental tab.",
        "suppressed_frustration": "Perform progressive muscle relaxation or step outside for a brisk 5-minute walk to release kinetic tension.",
        "somber_acceptance": "Recognize the resilience in your calm demeanor. Honor the moment of closure.",
        "sarcasm_or_detachment": "Humor is a valid defense mechanism, but check in with yourself to see what emotion sits just beneath it.",
        "authentic_joy": "Savor this positive mental state. Jot down what contributed to today's mood so you can recreate it.",
        "calm_contentment": "Notice the quiet clarity you feel right now. Great state for creative work or deep rest.",
        "authentic_sad": "Hydrate, wrap yourself in comfort, and refrain from harsh self-criticism today. Rest is productive.",
        "authentic_angry": "Channel intense energy into physical movement or structured problem-solving once your heart rate steadies.",
        "authentic_fearful": "Practice the 5-4-3-2-1 grounding exercise: notice 5 things you can see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "congruent_balanced": "A steady baseline today. Consistency in daily habits keeps your nervous system centered."
    }

    actionable_tip = tips.get(fused_emotion, tips.get(mismatch_type, "Notice your breath and take a mindful pause."))

    return {
        "fused_emotion": fused_emotion,
        "fused_valence": fused_valence,
        "fused_arousal": fused_arousal,
        "congruence_score": congruence_score,
        "is_mismatch": is_mismatch,
        "mismatch_type": mismatch_type,
        "mismatch_explanation": explanation,
        "actionable_tip": actionable_tip
    }
