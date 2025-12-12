"""FastAPI application for ArXiv Concept Tracker"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models import TrackingRequest, TrackingResponse, Paper
from backend.arxiv_client import ArXivClient
from backend.tracker import ConceptTracker
from backend.config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ArXiv Concept Tracker API",
    description="Track concept evolution through ArXiv papers using semantic embeddings and Kalman filtering",
    version="0.1.0"
)

# Add CORS middleware for frontend (future)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize services (singleton pattern for efficiency)
arxiv_client = ArXivClient()
concept_tracker = ConceptTracker()

logger.info("FastAPI application initialized")


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "ArXiv Concept Tracker API",
        "version": "0.1.0",
        "endpoints": {
            "/api/search": "Search ArXiv papers by keyword",
            "/api/track": "Track concept evolution from seed papers",
            "/api/paper/{arxiv_id}": "Get single paper details"
        },
        "docs": "/docs"
    }


@app.get("/api/search")
async def search_papers(query: str, limit: int = 20) -> dict:
    """
    Search ArXiv papers by keyword

    Args:
        query: Search query (supports ArXiv query syntax)
        limit: Maximum results to return (default: 20, max: 100)

    Returns:
        Dictionary with papers list and total count

    Example:
        GET /api/search?query=transformer&limit=10
    """
    if limit > 100:
        limit = 100

    logger.info(f"Search request: query='{query}', limit={limit}")

    try:
        papers = arxiv_client.search_papers(query, max_results=limit)
        logger.info(f"Search returned {len(papers)} papers")

        return {
            "papers": [p.model_dump() for p in papers],
            "total": len(papers),
            "query": query
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/track")
async def track_concept(request: TrackingRequest) -> TrackingResponse:
    """
    Track concept evolution from seed papers

    Args:
        request: TrackingRequest with seed paper IDs and parameters

    Returns:
        TrackingResponse with linear timeline of concept evolution

    Example:
        POST /api/track
        {
            "seed_paper_ids": ["1706.03762"],
            "end_date": "2018-12-31",
            "window_months": 6,
            "max_papers_per_window": 50
        }
    """
    logger.info(f"Track request: seeds={request.seed_paper_ids}, end={request.end_date}")

    try:
        response = concept_tracker.track(request)
        logger.info(f"Tracking complete: {response.num_steps} steps, {response.total_papers} papers")

        return response
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Tracking failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")


@app.get("/api/paper/{arxiv_id}")
async def get_paper(arxiv_id: str) -> Paper:
    """
    Get detailed information for a single paper

    Args:
        arxiv_id: ArXiv ID (e.g., "1706.03762")

    Returns:
        Paper object with metadata

    Example:
        GET /api/paper/1706.03762
    """
    logger.info(f"Get paper request: {arxiv_id}")

    try:
        paper = arxiv_client.get_paper_by_id(arxiv_id)
        logger.info(f"Paper retrieved: {paper.title[:50]}...")

        return paper
    except ValueError as e:
        logger.error(f"Paper not found: {arxiv_id}")
        raise HTTPException(status_code=404, detail=f"Paper not found: {arxiv_id}")
    except Exception as e:
        logger.error(f"Failed to fetch paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "arxiv-concept-tracker",
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
