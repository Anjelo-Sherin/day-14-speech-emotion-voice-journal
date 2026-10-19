import os
import shutil
import uuid
import email
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from src.config import settings
from src.database import get_db, init_db, JournalEntry
from src.audio_features import load_audio, compute_acoustic_prosody, extract_features_for_model
from src.model import predictor
from src.sentiment import analyze_text_sentiment
from src.multimodal_fusion import fuse_audio_and_text
from src.transcriber import transcriber
from src.reflective_summary import generate_reflective_summary
from src.sample_generator import generate_preset_samples, seed_database_if_empty, SAMPLE_PRESETS

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ensure database tables exist and seed demo data if empty on server start.
    """
    init_db()
    generate_preset_samples()
    seed_database_if_empty()
    print(f"[{settings.APP_NAME}] Startup complete. Server ready.")
    yield

# Initialize FastAPI application with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal Speech Emotion Recognition & Voice Journaling Platform",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = settings.BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount audio uploads
app.mount("/audio", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="audio")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>VocalMood backend is running. (static/index.html not found)</h1>")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/health")
def health_check():
    import torch
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": str(predictor.device),
        "pytorch_version": torch.__version__,
        "model_classes": predictor.emotions,
        "llm_gemini_configured": bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 10)
    }

@app.get("/api/samples")
def list_samples():
    """
    List interactive audio presets for one-click testing.
    """
    return [
        {
            "id": p["id"],
            "title": p["title"],
            "transcript": p["transcript"],
            "audio_url": f"/audio/sample_{p['id']}.wav"
        }
        for p in SAMPLE_PRESETS
    ]

def _extract_audio_and_transcript_from_request(body: bytes, content_type: str) -> tuple:
    """
    Parses audio bytes, filename, and client transcript from multipart, JSON, or octet stream.
    Zero external dependencies (uses Python standard library).
    """
    audio_bytes = None
    filename = "recording.wav"
    client_transcript = ""

    if "multipart/form-data" in content_type:
        msg = email.message_from_bytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        for part in msg.get_payload():
            name = part.get_param("name", header="content-disposition")
            if name == "file":
                audio_bytes = part.get_payload(decode=True)
                fn = part.get_filename()
                if fn:
                    filename = fn
            elif name == "client_transcript":
                client_transcript = part.get_payload(decode=True).decode("utf-8", errors="ignore")
    elif "application/json" in content_type:
        import json
        data = json.loads(body.decode("utf-8"))
        client_transcript = data.get("client_transcript", "")
        b64 = data.get("audio_base64", "")
        if b64:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(b64)
            filename = data.get("filename", "audio.wav")
    else:
        audio_bytes = body

    return audio_bytes, filename, client_transcript

@app.post("/api/analyze-audio")
async def analyze_audio(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    End-to-end multimodal pipeline:
    1. Save uploaded voice recording (WAV, WebM, etc.)
    2. Extract acoustic prosody & fixed MFCC tensors
    3. Run PyTorch CNN-LSTM inference for speech emotion
    4. Transcribe speech to text (Whisper / client speech recognition)
    5. Evaluate text sentiment & affective intent
    6. Fuse audio and text signals & flag vocal/verbal mismatches
    7. Generate empathetic reflective summary
    8. Store in SQLite database and return structured results
    """
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    audio_bytes, orig_filename, client_transcript = _extract_audio_and_transcript_from_request(body, content_type)

    if not audio_bytes or len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Invalid or empty audio payload received.")

    filename = f"{uuid.uuid4().hex}_{orig_filename}"
    file_path = settings.UPLOAD_DIR / filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {e}")

    try:
        y, sr = load_audio(file_path, target_sr=settings.SAMPLE_RATE)
        duration_sec = float(len(y) / sr)

        prosody = compute_acoustic_prosody(y, sr)

        feature_tensor = extract_features_for_model(
            y, sr,
            target_frames=settings.TARGET_FRAMES,
            n_mfcc=settings.N_MFCC
        )

        audio_prediction = predictor.predict(feature_tensor, prosody)
        final_transcript = transcriber.transcribe(file_path, client_transcript)
        text_prediction = analyze_text_sentiment(final_transcript)
        fusion_result = fuse_audio_and_text(audio_prediction, text_prediction)

        reflective_summary = generate_reflective_summary(
            transcript=final_transcript,
            audio_emotion=audio_prediction["dominant_emotion"],
            audio_confidence=audio_prediction["confidence"],
            text_sentiment=text_prediction["sentiment"],
            fused_emotion=fusion_result["fused_emotion"],
            congruence_score=fusion_result["congruence_score"],
            is_mismatch=fusion_result["is_mismatch"],
            mismatch_type=fusion_result["mismatch_type"]
        )

        entry = JournalEntry(
            created_at=datetime.utcnow(),
            audio_filename=filename,
            audio_path=str(file_path),
            duration_sec=duration_sec,
            transcript=final_transcript,
            audio_emotion=audio_prediction["dominant_emotion"],
            audio_confidence=audio_prediction["confidence"],
            audio_arousal=audio_prediction["audio_arousal"],
            audio_valence=audio_prediction["audio_valence"],
            text_sentiment=text_prediction["sentiment"],
            text_sentiment_score=text_prediction["score"],
            text_emotion=text_prediction["text_emotion"],
            fused_emotion=fusion_result["fused_emotion"],
            fused_valence=fusion_result["fused_valence"],
            fused_arousal=fusion_result["fused_arousal"],
            congruence_score=fusion_result["congruence_score"],
            is_mismatch=fusion_result["is_mismatch"],
            mismatch_type=fusion_result["mismatch_type"],
            mismatch_explanation=fusion_result["mismatch_explanation"],
            reflective_summary=reflective_summary,
            actionable_tip=fusion_result["actionable_tip"]
        )
        entry.emotion_breakdown = audio_prediction["emotion_breakdown"]

        db.add(entry)
        db.commit()
        db.refresh(entry)

        return {
            "success": True,
            "entry": entry.to_dict(),
            "prosody": prosody,
            "audio_url": f"/audio/{filename}"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Multimodal processing error: {str(e)}")

@app.post("/api/analyze-sample/{sample_id}")
async def analyze_preset_sample(sample_id: str, db: Session = Depends(get_db)):
    """
    Run pipeline on a curated demonstration sample.
    """
    matching = next((s for s in SAMPLE_PRESETS if s["id"] == sample_id), None)
    if not matching:
        raise HTTPException(status_code=404, detail="Preset sample not found")

    wav_path = settings.UPLOAD_DIR / f"sample_{sample_id}.wav"
    if not wav_path.exists():
        generate_preset_samples()

    y, sr = load_audio(wav_path, settings.SAMPLE_RATE)
    duration_sec = float(len(y) / sr)
    prosody = compute_acoustic_prosody(y, sr)
    feats = extract_features_for_model(y, sr, settings.TARGET_FRAMES, settings.N_MFCC)
    audio_prediction = predictor.predict(feats, prosody)

    text_prediction = analyze_text_sentiment(matching["transcript"])
    fusion_result = fuse_audio_and_text(audio_prediction, text_prediction)

    reflective_summary = generate_reflective_summary(
        transcript=matching["transcript"],
        audio_emotion=audio_prediction["dominant_emotion"],
        audio_confidence=audio_prediction["confidence"],
        text_sentiment=text_prediction["sentiment"],
        fused_emotion=fusion_result["fused_emotion"],
        congruence_score=fusion_result["congruence_score"],
        is_mismatch=fusion_result["is_mismatch"],
        mismatch_type=fusion_result["mismatch_type"]
    )

    entry = JournalEntry(
        created_at=datetime.utcnow(),
        audio_filename=wav_path.name,
        audio_path=str(wav_path),
        duration_sec=duration_sec,
        transcript=matching["transcript"],
        audio_emotion=audio_prediction["dominant_emotion"],
        audio_confidence=audio_prediction["confidence"],
        audio_arousal=audio_prediction["audio_arousal"],
        audio_valence=audio_prediction["audio_valence"],
        text_sentiment=text_prediction["sentiment"],
        text_sentiment_score=text_prediction["score"],
        text_emotion=text_prediction["text_emotion"],
        fused_emotion=fusion_result["fused_emotion"],
        fused_valence=fusion_result["fused_valence"],
        fused_arousal=fusion_result["fused_arousal"],
        congruence_score=fusion_result["congruence_score"],
        is_mismatch=fusion_result["is_mismatch"],
        mismatch_type=fusion_result["mismatch_type"],
        mismatch_explanation=fusion_result["mismatch_explanation"],
        reflective_summary=reflective_summary,
        actionable_tip=fusion_result["actionable_tip"]
    )
    entry.emotion_breakdown = audio_prediction["emotion_breakdown"]

    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {
        "success": True,
        "entry": entry.to_dict(),
        "prosody": prosody,
        "audio_url": f"/audio/{wav_path.name}"
    }

@app.get("/api/entries")
def get_entries(
    limit: int = Query(50, ge=1, le=200),
    mismatch_only: bool = False,
    db: Session = Depends(get_db)
):
    query = db.query(JournalEntry)
    if mismatch_only:
        query = query.filter(JournalEntry.is_mismatch == True)
    
    entries = query.order_by(desc(JournalEntry.created_at)).limit(limit).all()
    return [e.to_dict() for e in entries]

@app.delete("/api/entries/{entry_id}")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    
    if entry.audio_path and Path(entry.audio_path).exists() and "sample_" not in entry.audio_filename:
        try:
            os.remove(entry.audio_path)
        except Exception:
            pass

    db.delete(entry)
    db.commit()
    return {"success": True, "message": f"Entry {entry_id} deleted"}

@app.get("/api/trend")
def get_trend_data(days: int = Query(14, ge=3, le=60), db: Session = Depends(get_db)):
    start_date = datetime.utcnow() - timedelta(days=days)
    entries = (
        db.query(JournalEntry)
        .filter(JournalEntry.created_at >= start_date)
        .order_by(JournalEntry.created_at.asc())
        .all()
    )

    grouped: Dict[str, List[JournalEntry]] = {}
    for e in entries:
        d_str = e.created_at.strftime("%Y-%m-%d")
        if d_str not in grouped:
            grouped[d_str] = []
        grouped[d_str].append(e)

    trend_points = []
    emotion_counts: Dict[str, int] = {}
    total_mismatches = 0

    for i in range(days + 1):
        day_dt = start_date + timedelta(days=i)
        d_str = day_dt.strftime("%Y-%m-%d")
        day_entries = grouped.get(d_str, [])

        if day_entries:
            avg_val = sum(e.fused_valence for e in day_entries) / len(day_entries)
            avg_aro = sum(e.fused_arousal for e in day_entries) / len(day_entries)
            avg_congruence = sum(e.congruence_score for e in day_entries) / len(day_entries)
            mismatch_cnt = sum(1 for e in day_entries if e.is_mismatch)
            total_mismatches += mismatch_cnt

            for e in day_entries:
                emo = e.audio_emotion
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

            trend_points.append({
                "date": d_str,
                "label": day_dt.strftime("%b %d"),
                "valence": round(avg_val, 2),
                "arousal": round(avg_aro, 2),
                "congruence": round(avg_congruence, 1),
                "mismatch_count": mismatch_cnt,
                "entry_count": len(day_entries)
            })

    return {
        "trend_points": trend_points,
        "emotion_distribution": emotion_counts,
        "total_entries": len(entries),
        "total_mismatches": total_mismatches,
        "average_congruence": round(
            sum(e.congruence_score for e in entries) / max(1, len(entries)), 1
        )
    }

@app.get("/api/calendar-heatmap")
def get_calendar_heatmap(db: Session = Depends(get_db)):
    start_date = datetime.utcnow() - timedelta(days=35)
    entries = (
        db.query(JournalEntry)
        .filter(JournalEntry.created_at >= start_date)
        .order_by(JournalEntry.created_at.asc())
        .all()
    )

    day_map: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        d_str = e.created_at.strftime("%Y-%m-%d")
        if d_str not in day_map:
            day_map[d_str] = {
                "date": d_str,
                "valences": [],
                "emotions": [],
                "has_mismatch": False
            }
        day_map[d_str]["valences"].append(e.fused_valence)
        day_map[d_str]["emotions"].append(e.fused_emotion)
        if e.is_mismatch:
            day_map[d_str]["has_mismatch"] = True

    result = []
    for d_str, data in day_map.items():
        avg_v = sum(data["valences"]) / len(data["valences"])
        mode_emo = max(set(data["emotions"]), key=data["emotions"].count)
        result.append({
            "date": d_str,
            "average_valence": round(avg_v, 2),
            "dominant_emotion": mode_emo,
            "has_mismatch": data["has_mismatch"],
            "count": len(data["valences"])
        })
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
