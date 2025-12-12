"""Tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_search_endpoint():
    """Test search endpoint"""
    response = client.get("/api/search?query=transformer&limit=5")
    assert response.status_code == 200

    data = response.json()
    assert "papers" in data
    assert "total" in data
    assert len(data["papers"]) <= 5


def test_get_paper_endpoint():
    """Test get paper endpoint with known paper"""
    response = client.get("/api/paper/1706.03762")
    assert response.status_code == 200

    data = response.json()
    assert data["arxiv_id"] == "1706.03762"
    assert "attention" in data["title"].lower()


def test_get_paper_not_found():
    """Test get paper endpoint with invalid ID"""
    response = client.get("/api/paper/9999.99999")
    assert response.status_code == 404


@pytest.mark.slow
def test_track_endpoint():
    """Test tracking endpoint (slow - uses real ArXiv data)"""
    request_data = {
        "seed_paper_ids": ["1706.03762"],
        "end_date": "2017-12-31",
        "window_months": 6,
        "max_papers_per_window": 20
    }

    response = client.post("/api/track", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "timeline" in data
    assert "seed_papers" in data
    assert "total_papers" in data
    assert len(data["seed_papers"]) == 1
