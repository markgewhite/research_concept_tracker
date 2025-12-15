"""Pydantic models for ArXiv Concept Tracker"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class Paper(BaseModel):
    """Represents an ArXiv paper with metadata"""

    arxiv_id: str = Field(..., description="ArXiv ID (e.g., '1706.03762')")
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    updated: Optional[datetime] = None
    categories: list[str] = Field(default_factory=list, description="ArXiv categories (e.g., ['cs.LG', 'cs.CL'])")
    pdf_url: str

    # Computed fields (populated during tracking)
    embedding: Optional[list[float]] = Field(None, description="Embedding vector for paper")
    similarity: Optional[float] = Field(None, description="Similarity to concept vector")

    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "published": "2017-06-12T00:00:00",
                "categories": ["cs.CL", "cs.LG"],
                "pdf_url": "https://arxiv.org/pdf/1706.03762"
            }
        }


class TrackingStep(BaseModel):
    """Represents one step in the linear concept evolution timeline"""

    step_number: int = Field(..., description="Step index (1-based)")
    start_date: datetime = Field(..., description="Window start date")
    end_date: datetime = Field(..., description="Window end date")
    papers: list[Paper] = Field(..., description="Papers accepted in this window")
    concept_vector: list[float] = Field(..., description="Current concept embedding")
    velocity: list[float] = Field(..., description="Rate of concept change (Kalman)")
    avg_similarity: float = Field(..., description="Average similarity of papers to concept")
    num_high_confidence: int = Field(..., description="Number of papers with similarity >= 0.70")
    num_moderate: int = Field(..., description="Number of papers with similarity 0.60-0.70")
    num_low: int = Field(..., description="Number of papers with similarity < 0.60")

    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "start_date": "2017-06-12T00:00:00",
                "end_date": "2017-12-12T00:00:00",
                "papers": [],
                "concept_vector": [0.1, 0.2],
                "velocity": [0.01, 0.02],
                "avg_similarity": 0.78,
                "num_high_confidence": 12,
                "num_moderate": 8,
                "num_low": 3
            }
        }


class TrackingRequest(BaseModel):
    """Request to track concept evolution from seed papers"""

    seed_paper_ids: list[str] = Field(..., min_length=1, max_length=5,
                                       description="ArXiv IDs of seed papers (1-5)")
    end_date: str = Field(..., description="End date for tracking (ISO format: YYYY-MM-DD)")
    window_months: int = Field(6, gt=0, le=24,
                                description="Time window size in months (default: 6)")
    similarity_threshold: float = Field(0.65, ge=0.0, le=1.0,
                                        description="Minimum similarity threshold (default: 0.65)")
    max_papers_per_window: int = Field(50, gt=0, le=500,
                                        description="Max papers to fetch per window (default: 50)")

    class Config:
        json_schema_extra = {
            "example": {
                "seed_paper_ids": ["1706.03762"],
                "end_date": "2018-12-31",
                "window_months": 6,
                "similarity_threshold": 0.65,
                "max_papers_per_window": 50
            }
        }


class TrackingResponse(BaseModel):
    """Response containing the linear concept evolution timeline"""

    seed_papers: list[Paper] = Field(..., description="Seed papers used to initialize tracking")
    timeline: list[TrackingStep] = Field(..., description="Linear timeline of concept evolution")

    # Summary statistics
    total_papers: int = Field(..., description="Total papers tracked across all steps")
    num_steps: int = Field(..., description="Number of time steps")
    date_range: tuple[str, str] = Field(..., description="(start_date, end_date) ISO strings")

    # Tracking parameters used
    window_months: int
    similarity_threshold: float

    class Config:
        json_schema_extra = {
            "example": {
                "seed_papers": [],
                "timeline": [],
                "total_papers": 45,
                "num_steps": 3,
                "date_range": ("2017-06-12", "2018-12-31"),
                "window_months": 6,
                "similarity_threshold": 0.65
            }
        }
