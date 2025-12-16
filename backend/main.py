"""FastAPI application for ArXiv Concept Tracker"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Mount static files for frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    logger.info(f"Mounted frontend directory: {FRONTEND_DIR}")
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_DIR}")

logger.info("FastAPI application initialized")


@app.get("/")
def root():
    """Serve the frontend application"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        # Fallback to API information if frontend not available
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
async def search_papers(
    query: str,
    limit: int = 20,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> dict:
    """
    Search ArXiv papers by keyword with optional year range

    Args:
        query: Search query (supports ArXiv query syntax)
        limit: Maximum results to return (default: 20, max: 100)
        start_year: Optional start year (e.g., 2017)
        end_year: Optional end year (e.g., 2020)

    Returns:
        Dictionary with papers list and total count

    Example:
        GET /api/search?query=transformer&limit=10&start_year=2017&end_year=2018
    """
    if limit > 100:
        limit = 100

    # Convert years to datetime if provided
    start_date = datetime(start_year, 1, 1) if start_year else None
    end_date = datetime(end_year, 12, 31) if end_year else None

    logger.info(f"Search request: query='{query}', limit={limit}, years={start_year}-{end_year}")

    try:
        papers = arxiv_client.search_papers(
            query,
            start_date=start_date,
            end_date=end_date,
            max_results=limit
        )
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
