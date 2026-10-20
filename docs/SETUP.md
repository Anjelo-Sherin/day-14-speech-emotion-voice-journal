# Installation & Setup Guide

This guide provides instructions for setting up and running **VocalMood AI (Speech Emotion Recognition and Voice Journal)** locally or within a containerized environment.

---

## 1. System Prerequisites

- **Python:** Version 3.9, 3.10, or 3.11
- **Operating System:** macOS (Apple Silicon or Intel), Linux (Ubuntu 20.04+), or Windows 10/11 (WSL2 recommended)
- **Audio Codecs:** Standard OS audio subsystem (`libsndfile` / `ffmpeg` optional but recommended for full MP3/M4A support)
- **Hardware:** Standard CPU (runs inference in < 15ms); GPU/CUDA is automatically detected by PyTorch if available.

### Installing OS Audio Dependencies (Optional)
- **macOS (Homebrew):**
  ```bash
  brew install ffmpeg libsndfile
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
  ```

---

## 2. Local Python Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anjelo-Sherin/day-14-speech-emotion-voice-journal.git
cd day-14-speech-emotion-voice-journal
```

### Step 2: Create and Activate a Virtual Environment
```bash
python3 -m venv .venv

# On macOS / Linux:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to configure application preferences:
```ini
HOST=0.0.0.0
PORT=8014
DEBUG=True

# Optional: Google Gemini API for Reflective Wellness Summary
# If left empty, the built-in empathetic local synthesis engine runs automatically.
GEMINI_API_KEY=
```

---

## 4. Running the Application

### Start the FastAPI Server
```bash
python app.py
```

Or run via Uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8014 --reload
```

Once started, open your web browser at:
👉 **`http://localhost:8014`**

Interactive API documentation (Swagger UI) is available at:
👉 **`http://localhost:8014/docs`**

---

## 5. Docker Deployment

If you prefer running via Docker and Docker Compose:

### Using Docker Compose
```bash
docker compose up --build -d
```

### Checking Container Health
```bash
docker compose ps
docker compose logs -f
```

To stop the container:
```bash
docker compose down
```

---

## 6. Running Automated Tests

Run the complete test suite (unit tests for feature extraction, CNN-LSTM model, sentiment, fusion, and API routes):
```bash
pytest tests -v
```
