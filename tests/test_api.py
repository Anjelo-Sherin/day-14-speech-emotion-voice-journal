import pytest
from fastapi.testclient import TestClient
from app import app
from src.database import init_db
from src.sample_generator import generate_preset_samples

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_environment():
    init_db()
    generate_preset_samples()

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "pytorch_version" in data
    assert len(data["model_classes"]) == 8

def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 4
    sample_ids = [s["id"] for s in samples]
    assert "masked_distress" in sample_ids
    assert "authentic_joy" in sample_ids

def test_analyze_preset_sample():
    response = client.post("/api/analyze-sample/masked_distress")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "entry" in data
    assert "prosody" in data
    entry = data["entry"]
    assert entry["is_mismatch"] is True
    assert entry["mismatch_type"] == "masked_distress"
    assert entry["congruence_score"] < 70.0

def test_entries_and_trend_endpoints():
    res_entries = client.get("/api/entries?limit=10")
    assert res_entries.status_code == 200
    entries = res_entries.json()
    assert isinstance(entries, list)
    assert len(entries) > 0

    res_trend = client.get("/api/trend?days=14")
    assert res_trend.status_code == 200
    trend = res_trend.json()
    assert "trend_points" in trend
    assert "emotion_distribution" in trend
    assert trend["total_entries"] > 0

def test_calendar_heatmap():
    res_heatmap = client.get("/api/calendar-heatmap")
    assert res_heatmap.status_code == 200
    days = res_heatmap.json()
    assert isinstance(days, list)
    assert len(days) > 0
