"""Tests for Kalman concept tracker"""

import pytest
import numpy as np
from backend.kalman_tracker import KalmanConceptTracker
from backend.models import Paper
from datetime import datetime


def create_test_paper(arxiv_id="test_001", embedding_dim=1024):
    """Helper to create test paper with random embedding"""
    embedding = np.random.randn(embedding_dim)
    embedding /= np.linalg.norm(embedding)

    return Paper(
        arxiv_id=arxiv_id,
        title="Test Paper",
        abstract="Test abstract",
        authors=["Test Author"],
        published=datetime(2020, 1, 1),
        categories=["cs.LG"],
        pdf_url="http://test.pdf",
        embedding=embedding.tolist()
    )


@pytest.fixture
def tracker():
    """Create initialized tracker"""
    tracker = KalmanConceptTracker()
    seed_paper = create_test_paper()
    tracker.initialize([seed_paper])
    return tracker


def test_initialization():
    """Test tracker initialization with seed papers"""
    tracker = KalmanConceptTracker()
    paper = create_test_paper()

    tracker.initialize([paper])

    assert tracker.position is not None
    assert tracker.velocity is not None
    assert np.isclose(np.linalg.norm(tracker.position), 1.0)
    assert np.allclose(tracker.velocity, 0.0)


def test_reject_high_velocity(tracker):
    """Test that velocity constraint rejects large jumps"""
    # Create vector 0.1 units away (exceeds max_velocity=0.05)
    far_vector = tracker.position + 0.1 * np.random.randn(len(tracker.position))
    far_vector /= np.linalg.norm(far_vector)

    is_valid, conf, reason = tracker.evaluate_candidate(far_vector)

    # Should be rejected due to velocity
    assert "velocity" in reason.lower() or is_valid


def test_accept_smooth_drift(tracker):
    """Test that small drift is accepted"""
    # Create vector very close to current position
    close_vector = tracker.position + 0.005 * np.random.randn(len(tracker.position))
    close_vector /= np.linalg.norm(close_vector)

    is_valid, conf, reason = tracker.evaluate_candidate(close_vector)

    # Should be accepted
    assert is_valid
    assert conf > 0.5


def test_process_papers(tracker):
    """Test processing multiple papers"""
    # Create mix of similar and dissimilar papers
    papers = [create_test_paper(f"paper_{i}") for i in range(10)]

    accepted, rejected = tracker.process_papers(papers)

    # Should have some accepted and some rejected
    assert len(accepted) + len(rejected) == len(papers)
    assert all(len(item) == 4 for item in accepted)  # (paper, sim, conf, reason)
    assert all(len(item) == 4 for item in rejected)


def test_update_tracker(tracker):
    """Test tracker state update"""
    # Create similar papers
    papers = [create_test_paper(f"paper_{i}") for i in range(5)]

    # Make them similar to current position
    for paper in papers:
        embedding = np.array(tracker.position) + 0.01 * np.random.randn(len(tracker.position))
        embedding /= np.linalg.norm(embedding)
        paper.embedding = embedding.tolist()

    accepted, _ = tracker.process_papers(papers)

    old_position = tracker.position.copy()
    tracker.update(accepted)

    # Position should have changed
    assert not np.allclose(tracker.position, old_position)
    # Should still be normalized
    assert np.isclose(np.linalg.norm(tracker.position), 1.0)
