import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from src.config import settings

Base = declarative_base()

class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Audio metadata
    audio_filename = Column(String(255), nullable=True)
    audio_path = Column(String(512), nullable=True)
    duration_sec = Column(Float, default=0.0)
    
    # Transcript
    transcript = Column(Text, nullable=False, default="")
    
    # Audio Tone Analysis
    audio_emotion = Column(String(64), nullable=False, default="neutral")
    audio_confidence = Column(Float, default=0.0)
    audio_emotion_breakdown_json = Column(Text, default="{}")
    audio_arousal = Column(Float, default=0.5)
    audio_valence = Column(Float, default=0.0)
    
    # Text Sentiment Analysis
    text_sentiment = Column(String(32), nullable=False, default="neutral")
    text_sentiment_score = Column(Float, default=0.0)
    text_emotion = Column(String(64), default="neutral")
    
    # Multimodal Fusion
    fused_emotion = Column(String(64), nullable=False, default="neutral")
    fused_valence = Column(Float, default=0.0)
    fused_arousal = Column(Float, default=0.5)
    congruence_score = Column(Float, default=100.0)
    is_mismatch = Column(Boolean, default=False)
    mismatch_type = Column(String(64), default="congruent")
    mismatch_explanation = Column(Text, default="")
    
    # Wellness & Reflective Summary
    reflective_summary = Column(Text, default="")
    actionable_tip = Column(Text, default="")

    @property
    def emotion_breakdown(self) -> Dict[str, float]:
        try:
            return json.loads(self.audio_emotion_breakdown_json or "{}")
        except Exception:
            return {}

    @emotion_breakdown.setter
    def emotion_breakdown(self, value: Dict[str, float]):
        self.audio_emotion_breakdown_json = json.dumps(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "audio_filename": self.audio_filename,
            "duration_sec": round(self.duration_sec, 2),
            "transcript": self.transcript,
            "audio_emotion": self.audio_emotion,
            "audio_confidence": round(self.audio_confidence, 4),
            "audio_emotion_breakdown": self.emotion_breakdown,
            "audio_arousal": round(self.audio_arousal, 3),
            "audio_valence": round(self.audio_valence, 3),
            "text_sentiment": self.text_sentiment,
            "text_sentiment_score": round(self.text_sentiment_score, 3),
            "text_emotion": self.text_emotion,
            "fused_emotion": self.fused_emotion,
            "fused_valence": round(self.fused_valence, 3),
            "fused_arousal": round(self.fused_arousal, 3),
            "congruence_score": round(self.congruence_score, 1),
            "is_mismatch": self.is_mismatch,
            "mismatch_type": self.mismatch_type,
            "mismatch_explanation": self.mismatch_explanation,
            "reflective_summary": self.reflective_summary,
            "actionable_tip": self.actionable_tip,
        }

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
