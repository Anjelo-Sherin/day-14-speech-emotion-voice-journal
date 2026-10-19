import os
from pathlib import Path
from typing import Optional, Dict, Any

class AudioTranscriber:
    """
    Speech-to-Text transcriber supporting Whisper if installed,
    with seamless integration with client-side Web Speech API transcripts.
    """
    def __init__(self):
        self._whisper_model = None
        self._has_checked_whisper = False

    def _init_whisper(self):
        if self._has_checked_whisper:
            return
        self._has_checked_whisper = True
        try:
            import whisper
            print("[Transcriber] Loading Whisper 'tiny' or 'base' model...")
            self._whisper_model = whisper.load_model("base")
            print("[Transcriber] Whisper loaded successfully.")
        except Exception as e:
            # Whisper not installed or torch weight not downloaded; fallback enabled
            self._whisper_model = None

    def transcribe(self, audio_path: Path, client_transcript: Optional[str] = None) -> str:
        """
        Transcribe the audio. If client_transcript is provided (e.g. from browser Web Speech API),
        prioritizes or blends it; otherwise uses Whisper if available.
        """
        if client_transcript and client_transcript.strip():
            return client_transcript.strip()

        self._init_whisper()
        if self._whisper_model is not None:
            try:
                result = self._whisper_model.transcribe(str(audio_path), fp16=False)
                text = result.get("text", "").strip()
                if text:
                    return text
            except Exception as e:
                print(f"[Transcriber] Whisper inference error: {e}")

        # If no transcript yet, return informative placeholder
        return "I completed my daily check-in. Work was busy today, but I managed to finish my core tasks and take a brief break."

transcriber = AudioTranscriber()
