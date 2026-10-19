# VocalMood: Technical Architecture & System Design

> **Project 14 of 30:** Speech Emotion Recognition and Voice Journal  
> **Domain:** Audio Machine Learning + Natural Language Processing + Affective Computing

---

## 1. System Architecture Overview

VocalMood is an end-to-end multimodal affective computing platform designed for privacy-first voice journaling. Rather than relying solely on lexical transcripts (which frequently conceal emotional distress beneath socially pleasant words), VocalMood extracts and fuses two orthogonal modalities:
1. **Acoustic Vocal Prosody:** Spectro-temporal features capturing pitch ($F_0$), harmonic energy, zero-crossing rate, and 40 MFCCs processed through a deep **PyTorch CNN-LSTM with Temporal Attention**.
2. **Lexical Semantic Sentiment:** Syntactic and emotional intent analysis of transcribed speech, accounting for negation, intensifiers, and emotional masking phrases (e.g., *"I'm totally fine"*).

```
 ┌────────────────────────┐
 │ Browser Audio Capture  │
 │ (MediaRecorder / File) │
 └───────────┬────────────┘
             │
      ┌──────┴─────────────────────────┐
      ▼                                ▼
┌─────────────────────────┐   ┌────────────────────────┐
│ Acoustic Preprocessing  │   │  Speech-to-Text (STT)  │
│ - Mel-Spectrogram (64)  │   │  (Whisper / Web Speech)│
│ - 40 MFCCs + Prosody    │   └───────────┬────────────┘
│ - F0 Pitch + RMS + ZCR  │               │
└────────────┬────────────┘               ▼
             ▼                ┌────────────────────────┐
┌─────────────────────────┐   │ Text Sentiment Engine  │
│ PyTorch CNN-LSTM + Attn │   │ - Lexical Valence      │
│ (8-Class Emotion Probs) │   │ - Negation & Masking   │
└────────────┬────────────┘   └───────────┬────────────┘
             │                            │
             └─────────────┬──────────────┘
                           ▼
             ┌─────────────────────────┐
             │ Multimodal Fusion Layer │
             │ - Affective Congruence  │
             │ - Mismatch Classifier   │
             │ - Fused Valence/Arousal │
             └─────────────┬──────────────┘
                           ▼
             ┌─────────────────────────┐
             │ Reflective Wellness LLM │
             │ (Gemini API / Empathetic│
             │  Local Synthesis Engine)│
             └─────────────┬──────────────┘
                           ▼
             ┌─────────────────────────┐
             │ SQLite / Dashboard UI   │
             │ - Calendar Heatmap      │
             │ - 14-Day Trajectory     │
             │ - Audio Player & Export │
             └─────────────────────────┘
```

---

## 2. Acoustic Feature Extraction Pipeline

1. **Downsampling & Normalization:** Standardizes raw PCM audio to 16 kHz mono float32 in $[-1.0, 1.0]$.
2. **Mel-Filterbank Decomposition:** Applies a 1024-point STFT with a 512-hop Hann window through a 64-band triangular Mel scale spanning 0 Hz to 8000 Hz.
3. **MFCC Computation:** Applies Discrete Cosine Transform (DCT Type-II) over log Mel energies to isolate 40 cepstral coefficients, capturing the acoustic transfer function of the vocal tract independent of pitch.
4. **Prosodic Descriptors:**
   - **Root Mean Square (RMS) Energy:** Measures vocal projection and dynamics.
   - **Zero Crossing Rate (ZCR):** Gauges vocal friction and agitation.
   - **Autocorrelation Pitch ($F_0$):** Tracks fundamental vocal fold frequency in the 50–400 Hz range to gauge physiological arousal.
   - **Spectral Centroid:** Tracks spectral brightness (higher centroid = brighter/tense voice).

---

## 3. PyTorch CNN-LSTM Neural Network

```
Input: Tensor of shape (Batch, 1, 40 MFCCs, 128 Time Frames)
  │
  ├── 2D Convolution (32 filters, 3x3, BatchNorm, ReLU) ──► MaxPool(2,2) ──► Dropout(0.2)
  ├── 2D Convolution (64 filters, 3x3, BatchNorm, ReLU) ──► MaxPool(2,2) ──► Dropout(0.2)
  ├── 2D Convolution (128 filters, 3x3, BatchNorm, ReLU) ─► MaxPool(2,1) ──► Dropout(0.2)
  │
  ├── Spatial Flattening ──► Reshape to (Batch, 16 Time Steps, 640 Features)
  │
  ├── Bidirectional LSTM (2 Layers, Hidden Dim = 64, Dropout = 0.2)
  │   └── Captures bidirectional prosodic trajectories across the voice clip
  │
  ├── Learned Temporal Attention Layer
  │   ├── α_t = Softmax(W2 · Tanh(W1 · h_t + b1) + b2)
  │   └── Context Vector = ∑ (α_t · h_t)
  │
  └── Dense Classifier (Linear 128 -> 64 -> ReLU -> Dropout 0.3 -> Linear 64 -> 8 Classes)
      └── Outputs Calibrated Logits for:
          [Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised]
```

---

## 4. Multimodal Fusion & Affective Mismatch Logic

Affective congruence represents the concordance between the semantic content of speech and involuntary vocal acoustics:

$$\text{Valence}_{\text{fused}} = 0.60 \cdot V_{\text{audio}} + 0.40 \cdot V_{\text{text}}$$

$$\text{Congruence Score} = \max(5.0, 100.0 - 45.0 \cdot |V_{\text{audio}} - V_{\text{text}}| - \text{Penalties})$$

### Mismatch Decision Matrix
| Verbal Sentiment | Vocal Tone Emotion | Classification | Clinical/Wellness Interpretation |
|---|---|---|---|
| Positive / Cheerful | Sad / Fearful / Low Energy | **Masked Distress** | Masking vulnerability; putting on a brave face |
| Composed / Neutral | Angry / High Arousal | **Suppressed Frustration** | Repressed irritation or agitation |
| Somber / Heavy | Calm / Serene / Low Arousal | **Somber Acceptance** | Resilience, closure, or peaceful processing |
| Hyper-Positive | Disgust / Detached Flat | **Sarcasm / Irony** | Cynicism, self-deprecating defense mechanism |
| Positive | Warm / Energetic Happy | **Congruent Joy** | Authentic positive affect |

---

## 5. Interview Talking Points & Deep-Dive Questions

### Q1: Why fuse acoustic tone with transcript text rather than relying on Whisper + an LLM alone?
*Answer:* Transcripts only capture *what* was said, missing physiological leakage (vocal micro-tremors, pitch contour, acoustic energy). In human emotional communication, Mehrabian’s communication model notes that tone conveys substantial emotional intent. An LLM reading *"I'm fine, really fine"* marks it positive; our multimodal fusion detects low pitch variance, subdued RMS energy, and acoustic sadness, flagging **Masked Distress**.

### Q2: How does the dataset bias in RAVDESS affect real-world generalization?
*Answer:* The RAVDESS dataset consists of professional actors portraying emotions at normal and strong intensity levels in quiet studio environments. In the wild, everyday speech features spontaneous micro-expressions, overlapping emotions, background noise, and varying room acoustics. To mitigate this:
1. We compute invariant normalized MFCCs with per-utterance cepstral mean/variance normalization (CMVN).
2. We supplement neural features with robust prosodic physics (autocorrelation pitch contours, spectral centroids, RMS).
3. We employ temporal self-attention to focus on expressive frames rather than silent pauses.

### Q3: How do you address privacy and biometric data ethics?
*Answer:* Voice samples contain sensitive biometric signatures. VocalMood is built with **privacy-by-design**:
- Audio files and SQLite database entries remain local to the user's host environment.
- The CNN-LSTM and sentiment classifiers execute locally on CPU/GPU without sending raw voice streams to third-party cloud servers.
- Clear wellness aid disclaimers prevent user misinterpretation as a medical diagnosis.
