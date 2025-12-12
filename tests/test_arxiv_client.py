"""Tests for ArXiv client"""

import pytest
from backend.arxiv_client import ArXivClient


@pytest.fixture
def client():
    """Create ArXiv client for tests"""
    return ArXivClient()


def test_get_transformer_paper(client):
    """Test fetching the famous transformer paper"""
    paper = client.get_paper_by_id("1706.03762")

    assert paper.arxiv_id == "1706.03762"
    assert "attention" in paper.title.lower()
    assert paper.published.year == 2017
    assert len(paper.authors) > 0
    assert paper.pdf_url.startswith("http")


def test_get_paper_not_found(client):
    """Test fetching non-existent paper raises error"""
    with pytest.raises(ValueError, match="not found"):
        client.get_paper_by_id("9999.99999")


def test_search_papers(client):
    """Test searching for papers"""
    papers = client.search_papers("attention is all you need", max_results=5)

    assert len(papers) <= 5
    assert len(papers) > 0
    assert all(hasattr(p, "title") for p in papers)
    assert all(hasattr(p, "arxiv_id") for p in papers)


def test_get_papers_by_ids(client):
    """Test batch fetching papers"""
    ids = ["1706.03762", "1810.04805"]  # Transformer and BERT papers
    papers = client.get_papers_by_ids(ids)

    assert len(papers) == 2
    assert papers[0].arxiv_id == "1706.03762"
    assert papers[1].arxiv_id == "1810.04805"
