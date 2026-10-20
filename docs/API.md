# VocalMood REST API Documentation

VocalMood exposes an asynchronous RESTful API powered by FastAPI for audio analysis, mood trajectory tracking, and journal management.

---

## 1. Health & System Info

### `GET /api/health`
Returns system status, device info, PyTorch version, and model capabilities.

**Sample Request:**
```bash
curl -X GET http://localhost:8014/api/health
```

**Sample Response:**
```json
{
  "status": "healthy",
  "service": "VocalMood: Speech Emotion & Voice Journal",
  "version": "1.0.0",
  "device": "cpu",
  "pytorch_version": "2.8.0",
  "model_classes": [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"
  ],
  "llm_gemini_configured": false
}
```

---

## 2. Multimodal Audio Analysis

### `POST /api/analyze-audio`
Uploads and processes an audio file (WAV, WebM, MP3) with optional client transcript.

**Supported Content-Types:**
- `multipart/form-data`: `file` (audio file), `client_transcript` (optional string)
- `application/json`: `{ "audio_base64": "...", "client_transcript": "...", "filename": "audio.wav" }`

**Sample cURL (Multipart):**
```bash
curl -X POST http://localhost:8014/api/analyze-audio \
  -F "file=@sample.wav" \
  -F "client_transcript=Everything is going well today."
```

**Response (`200 OK`):**
```json
{
  "success": true,
  "entry": {
    "id": 15,
    "created_at": "2026-10-20T08:30:00",
    "audio_filename": "f29b478_sample.wav",
    "duration_sec": 4.2,
    "transcript": "Everything is going well today.",
    "audio_emotion": "sad",
    "audio_confidence": 0.82,
    "audio_arousal": 0.35,
    "audio_valence": -0.42,
    "text_sentiment": "positive",
    "text_sentiment_score": 0.54,
    "fused_emotion": "suppressed_distress",
    "fused_valence": -0.04,
    "fused_arousal": 0.45,
    "congruence_score": 38.5,
    "is_mismatch": true,
    "mismatch_type": "masked_distress",
    "mismatch_explanation": "Vocal tone reveals underlying sad strain despite your positive words.",
    "reflective_summary": "You spoke reassuring words about your day, but your vocal acoustics carried unmistakable traces of fatigue...",
    "actionable_tip": "Write down the single biggest friction point right now on paper, then close the notebook."
  },
  "prosody": {
    "rms_mean": 0.052,
    "pitch_mean_hz": 122.4,
    "pitch_std_hz": 7.2,
    "spectral_centroid_mean": 1840.5,
    "estimated_arousal": 0.38,
    "estimated_valence": -0.35
  },
  "audio_url": "/audio/f29b478_sample.wav"
}
```

---

## 3. Demonstration Presets

### `GET /api/samples`
Returns list of built-in audio samples for instant 1-click evaluation.

**Sample Request:**
```bash
curl -X GET http://localhost:8014/api/samples
```

### `POST /api/analyze-sample/{sample_id}`
Executes the multimodal pipeline on a curated sample without needing a microphone.

**Sample IDs:**
- `masked_distress`: Positive words ("Everything is fine") with depressed/strained vocal tone.
- `authentic_joy`: Cheerful words ("We crossed our milestone") with bright, animated voice.
- `mindful_calm`: Peaceful reflection with steady harmonic baseline.
- `suppressed_frustration`: Composed words with tense, high-arousal acoustic bursts.

**Sample Request:**
```bash
curl -X POST http://localhost:8014/api/analyze-sample/masked_distress
```

---

## 4. Trajectory & Calendar Analytics

### `GET /api/trend?days=14`
Aggregates daily emotional valence, acoustic arousal, and congruence score.

**Sample Request:**
```bash
curl -X GET "http://localhost:8014/api/trend?days=14"
```

### `GET /api/calendar-heatmap`
Returns a 30-day daily breakdown of mood intensities and mismatch occurrences for calendar heatmap rendering.

**Sample Request:**
```bash
curl -X GET http://localhost:8014/api/calendar-heatmap
```

---

## 5. Journal History Management

### `GET /api/entries?limit=50&mismatch_only=false`
Retrieves past voice journal entries ordered by timestamp descending.

### `DELETE /api/entries/{entry_id}`
Deletes a specific journal entry and cleans up associated audio storage.
