// VocalMood Voice Journal Client-Side Application

let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let visualizerAnimationId = null;
let recordingStartTime = null;
let timerInterval = null;
let speechRecognizer = null;

let radarChart = null;
let trendChart = null;
let currentFilterMismatchOnly = false;

document.addEventListener('DOMContentLoaded', () => {
  initVisualizer();
  initCharts();
  initSpeechRecognition();
  loadPresets();
  loadTrendData();
  loadCalendarHeatmap();
  loadHistoryEntries();

  document.getElementById('micBtn').addEventListener('click', toggleRecording);
});

// ----------------------------------------------------
// 1. Audio Visualizer Canvas
// ----------------------------------------------------
function initVisualizer() {
  const canvas = document.getElementById('visualizer');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth || 300;
  canvas.height = canvas.offsetHeight || 60;

  // Draw idle baseline
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'rgba(139, 92, 246, 0.4)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

function startVisualizerStream(stream) {
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const canvas = document.getElementById('visualizer');
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      visualizerAnimationId = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#8b5cf6';
      ctx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    }

    draw();
  } catch (err) {
    console.warn("Visualizer audio context unavailable:", err);
  }
}

function stopVisualizer() {
  if (visualizerAnimationId) {
    cancelAnimationFrame(visualizerAnimationId);
    visualizerAnimationId = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close();
  }
  initVisualizer();
}

// ----------------------------------------------------
// 2. Web Speech Recognition (Browser STT)
// ----------------------------------------------------
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = 'en-US';

    speechRecognizer.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const input = document.getElementById('transcriptInput');
      if (finalTranscript) {
        input.value = (input.value + ' ' + finalTranscript).trim();
      } else if (interimTranscript) {
        input.placeholder = interimTranscript;
      }
    };

    speechRecognizer.onerror = (e) => {
      console.warn("Speech recognition error:", e.error);
    };
  }
}

// ----------------------------------------------------
// 3. Audio Recording Handling
// ----------------------------------------------------
async function toggleRecording() {
  const micBtn = document.getElementById('micBtn');
  const recordStatus = document.getElementById('recordStatus');

  if (mediaRecorder && mediaRecorder.state === 'recording') {
    // Stop recording
    mediaRecorder.stop();
    micBtn.classList.remove('recording');
    recordStatus.textContent = 'Processing voice audio...';
    clearInterval(timerInterval);
    stopVisualizer();

    if (speechRecognizer) {
      try { speechRecognizer.stop(); } catch (e) {}
    }
  } else {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        stream.getTracks().forEach(track => track.stop());
        await uploadAndAnalyzeAudio(audioBlob);
      };

      mediaRecorder.start();
      micBtn.classList.add('recording');
      recordStatus.textContent = 'Recording live... Speak naturally.';

      startVisualizerStream(stream);

      // Start timer
      recordingStartTime = Date.now();
      timerInterval = setInterval(updateTimer, 1000);

      if (speechRecognizer) {
        try { speechRecognizer.start(); } catch (e) {}
      }
    } catch (err) {
      alert("Microphone access was denied or is unavailable. You can use preset samples or upload an audio file below!");
      recordStatus.textContent = 'Microphone access failed. Try preset samples below.';
    }
  }
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
  const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const secs = String(elapsed % 60).padStart(2, '0');
  document.getElementById('recordingTimer').textContent = `${mins}:${secs}`;
}

async function uploadAndAnalyzeAudio(blob) {
  const recordStatus = document.getElementById('recordStatus');
  recordStatus.textContent = 'Extracting MFCCs & running CNN-LSTM inference...';

  const formData = new FormData();
  formData.append('file', blob, 'user_recording.webm');
  const transcriptVal = document.getElementById('transcriptInput').value.trim();
  if (transcriptVal) {
    formData.append('client_transcript', transcriptVal);
  }

  try {
    const res = await fetch('/api/analyze-audio', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Analysis error');
    }
    const data = await res.json();
    renderAnalysisResult(data.entry, data.audio_url);
    recordStatus.textContent = 'Voice note analyzed & saved to journal!';
    refreshDashboard();
  } catch (err) {
    alert("Error analyzing audio: " + err.message);
    recordStatus.textContent = 'Analysis error occurred.';
  }
}

async function handleFileUpload() {
  const fileInput = document.getElementById('audioFileInput');
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select an audio file first (.wav, .mp3, etc.)");
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  const transcriptVal = document.getElementById('transcriptInput').value.trim();
  if (transcriptVal) {
    formData.append('client_transcript', transcriptVal);
  }

  const recordStatus = document.getElementById('recordStatus');
  recordStatus.textContent = 'Uploading & processing audio file...';

  try {
    const res = await fetch('/api/analyze-audio', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Processing failed');
    renderAnalysisResult(data.entry, data.audio_url);
    recordStatus.textContent = 'File analyzed successfully!';
    refreshDashboard();
  } catch (err) {
    alert("Upload error: " + err.message);
    recordStatus.textContent = 'File analysis error.';
  }
}

// ----------------------------------------------------
// 4. Presets Handling
// ----------------------------------------------------
async function loadPresets() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();
    const container = document.getElementById('presetPillsContainer');
    container.innerHTML = '';

    samples.forEach(sample => {
      const pill = document.createElement('button');
      pill.className = 'preset-pill';
      pill.innerHTML = `<span>▶</span> ${sample.title}`;
      pill.onclick = () => runPresetSample(sample.id);
      container.appendChild(pill);
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

async function runPresetSample(sampleId) {
  const recordStatus = document.getElementById('recordStatus');
  recordStatus.textContent = `Analyzing preset "${sampleId}"...`;

  try {
    const res = await fetch(`/api/analyze-sample/${sampleId}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Preset error');

    // Populate transcript box with the sample text
    document.getElementById('transcriptInput').value = data.entry.transcript;

    renderAnalysisResult(data.entry, data.audio_url);
    recordStatus.textContent = `Preset "${data.entry.fused_emotion}" loaded & analyzed!`;
    refreshDashboard();
  } catch (err) {
    alert("Failed to run preset: " + err.message);
  }
}

// ----------------------------------------------------
// 5. Render Multimodal Analysis Results
// ----------------------------------------------------
function renderAnalysisResult(entry, audioUrl) {
  // 1. Audio playback
  if (audioUrl) {
    const playerContainer = document.getElementById('playerContainer');
    const player = document.getElementById('audioPlayback');
    player.src = audioUrl;
    playerContainer.style.display = 'block';
  }

  // 2. Fused state badge
  const fusedBadge = document.getElementById('fusedMoodBadge');
  fusedBadge.textContent = entry.fused_emotion.replace('_', ' ').toUpperCase();
  fusedBadge.className = `badge ${entry.is_mismatch ? 'badge-mismatch' : 'badge-happy'}`;

  // 3. Congruence Meter
  const congVal = document.getElementById('congruenceValue');
  const congBar = document.getElementById('congruenceBar');
  congVal.textContent = `${entry.congruence_score}%`;
  congBar.style.width = `${entry.congruence_score}%`;
  if (entry.congruence_score < 60) {
    congBar.style.background = 'var(--accent-danger)';
  } else if (entry.congruence_score < 80) {
    congBar.style.background = 'var(--accent-warning)';
  } else {
    congBar.style.background = 'var(--accent-success)';
  }

  // 4. Mismatch banner
  const mismatchBox = document.getElementById('mismatchBox');
  const mismatchTitle = document.getElementById('mismatchTitle');
  const mismatchDesc = document.getElementById('mismatchDesc');
  const mismatchIcon = document.getElementById('mismatchIcon');

  if (entry.is_mismatch) {
    mismatchBox.style.display = 'flex';
    mismatchBox.className = 'mismatch-alert';
    mismatchIcon.textContent = '⚡';
    mismatchTitle.textContent = `Mismatch Detected: ${entry.mismatch_type.replace('_', ' ').toUpperCase()}`;
    mismatchDesc.textContent = entry.mismatch_explanation;
  } else {
    mismatchBox.style.display = 'flex';
    mismatchBox.className = 'mismatch-alert congruent';
    mismatchIcon.textContent = '✨';
    mismatchTitle.textContent = 'Affective Harmony: Words & Voice Aligned';
    mismatchDesc.textContent = entry.mismatch_explanation || 'Your tone and lexical sentiment are in congruence.';
  }

  // 5. Tone & Sentiment details
  document.getElementById('acousticToneLabel').textContent = entry.audio_emotion;
  document.getElementById('acousticToneConf').textContent = `${(entry.audio_confidence * 100).toFixed(0)}%`;
  document.getElementById('textSentimentLabel').textContent = entry.text_sentiment;
  document.getElementById('textSentimentScore').textContent = `Valence: ${entry.text_sentiment_score > 0 ? '+' : ''}${entry.text_sentiment_score}`;

  document.getElementById('arousalVal').textContent = entry.fused_arousal;
  document.getElementById('valenceVal').textContent = entry.fused_valence;

  // 6. Update Radar Chart
  updateRadarChart(entry.audio_emotion_breakdown || {});

  // 7. Reflective Summary & Micro-Habit
  document.getElementById('reflectiveSummaryText').textContent = entry.reflective_summary;
  document.getElementById('actionTipText').textContent = entry.actionable_tip || 'Take a deep breath and stay hydrated.';
}

// ----------------------------------------------------
// 6. Chart.js Initializations & Updates
// ----------------------------------------------------
function initCharts() {
  // Emotion Radar
  const radarCtx = document.getElementById('emotionRadarChart').getContext('2d');
  radarChart = new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised'],
      datasets: [{
        label: 'Acoustic Tone Probability',
        data: [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
        backgroundColor: 'rgba(139, 92, 246, 0.25)',
        borderColor: '#8b5cf6',
        pointBackgroundColor: '#c4b5fd',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: '#8b5cf6'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: {
            color: '#cbd5e1',
            font: { size: 10, family: 'Inter' }
          },
          ticks: { display: false, maxTicksLimit: 5 },
          suggestedMin: 0,
          suggestedMax: 0.6
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // Weekly Trend Chart
  const trendCtx = document.getElementById('weeklyTrendChart').getContext('2d');
  trendChart = new Chart(trendCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Fused Valence (-1 to +1)',
          data: [],
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          tension: 0.35,
          fill: true,
          pointRadius: 4
        },
        {
          label: 'Acoustic Arousal (0 to 1)',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'transparent',
          borderDash: [4, 4],
          tension: 0.35,
          pointRadius: 3
        },
        {
          label: 'Congruence % (/100)',
          data: [],
          borderColor: '#8b5cf6',
          backgroundColor: 'transparent',
          tension: 0.35,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        },
        y: {
          min: -1.0,
          max: 1.0,
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          ticks: { color: '#94a3b8', font: { size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#e2e8f0', font: { size: 11 } }
        }
      }
    }
  });
}

function updateRadarChart(breakdown) {
  if (!radarChart) return;
  const labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'];
  const data = labels.map(l => breakdown[l] || 0.02);
  radarChart.data.datasets[0].data = data;
  radarChart.update();
}

async function loadTrendData() {
  try {
    const res = await fetch('/api/trend?days=14');
    const data = await res.json();
    if (!trendChart || !data.trend_points) return;

    trendChart.data.labels = data.trend_points.map(p => p.label);
    trendChart.data.datasets[0].data = data.trend_points.map(p => p.valence);
    trendChart.data.datasets[1].data = data.trend_points.map(p => p.arousal);
    trendChart.data.datasets[2].data = data.trend_points.map(p => p.congruence / 100.0);
    trendChart.update();
  } catch (err) {
    console.error("Error loading trend data:", err);
  }
}

// ----------------------------------------------------
// 7. Calendar Heatmap
// ----------------------------------------------------
async function loadCalendarHeatmap() {
  try {
    const res = await fetch('/api/calendar-heatmap');
    const heatmapDays = await res.json();
    const grid = document.getElementById('calendarHeatmapGrid');
    grid.innerHTML = '';

    heatmapDays.forEach(day => {
      const tile = document.createElement('div');
      tile.className = 'heatmap-tile';

      // Valence intensity class
      let level = 1;
      if (day.average_valence > 0.4) level = 4;
      else if (day.average_valence > 0.15) level = 3;
      else if (day.average_valence > -0.15) level = 2;
      else level = 1;

      tile.classList.add(`tile-level-${level}`);
      if (day.has_mismatch) {
        tile.classList.add('tile-mismatch');
      }

      tile.title = `${day.date}: Mood ${day.dominant_emotion} (Valence: ${day.average_valence})${day.has_mismatch ? ' [Mismatch Detected]' : ''}`;
      grid.appendChild(tile);
    });
  } catch (err) {
    console.error("Error loading calendar heatmap:", err);
  }
}

// ----------------------------------------------------
// 8. History Feed
// ----------------------------------------------------
function setHistoryFilter(mismatchOnly) {
  currentFilterMismatchOnly = mismatchOnly;
  document.getElementById('filterAllBtn').style.background = mismatchOnly ? 'transparent' : 'rgba(255,255,255,0.15)';
  document.getElementById('filterMismatchBtn').style.background = mismatchOnly ? 'rgba(239,68,68,0.25)' : 'transparent';
  loadHistoryEntries();
}

async function loadHistoryEntries() {
  try {
    const url = `/api/entries?limit=30${currentFilterMismatchOnly ? '&mismatch_only=true' : ''}`;
    const res = await fetch(url);
    const entries = await res.json();
    const feed = document.getElementById('historyFeed');
    feed.innerHTML = '';

    if (entries.length === 0) {
      feed.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem; padding: 1rem 0;">No journal entries found.</p>';
      return;
    }

    entries.forEach(entry => {
      const item = document.createElement('div');
      item.className = 'history-item';

      const dateStr = new Date(entry.created_at).toLocaleString([], {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
      });

      item.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="display: flex; gap: 0.5rem; align-items: center;">
            <span class="badge ${entry.is_mismatch ? 'badge-mismatch' : 'badge-calm'}">
              ${entry.is_mismatch ? '⚡ ' : ''}${entry.fused_emotion.replace('_', ' ')}
            </span>
            <span style="font-size: 0.78rem; color: var(--text-muted);">${dateStr}</span>
          </div>
          <div style="display: flex; gap: 0.5rem; align-items: center;">
            <span style="font-size: 0.75rem; color: var(--text-muted);">Congruence: ${entry.congruence_score}%</span>
            <button class="btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;" onclick="deleteEntry(${entry.id})">✕</button>
          </div>
        </div>
        <p style="font-size: 0.88rem; color: #e2e8f0; font-style: italic;">"${entry.transcript}"</p>
        <div style="font-size: 0.82rem; color: #cbd5e1; background: rgba(15,23,42,0.4); padding: 0.6rem; border-radius: 6px;">
          ${entry.reflective_summary}
        </div>
        ${entry.audio_filename ? `<audio controls style="height: 30px; margin-top: 0.3rem;" src="/audio/${entry.audio_filename}"></audio>` : ''}
      `;
      feed.appendChild(item);
    });
  } catch (err) {
    console.error("Error loading entries:", err);
  }
}

async function deleteEntry(entryId) {
  if (!confirm("Delete this journal entry?")) return;
  try {
    const res = await fetch(`/api/entries/${entryId}`, { method: 'DELETE' });
    if (res.ok) {
      refreshDashboard();
    }
  } catch (err) {
    alert("Failed to delete entry: " + err.message);
  }
}

function refreshDashboard() {
  loadTrendData();
  loadCalendarHeatmap();
  loadHistoryEntries();
}

async function exportData(format) {
  try {
    const res = await fetch('/api/entries?limit=100');
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voice_journal_export_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  } catch (err) {
    alert("Export failed: " + err.message);
  }
}
