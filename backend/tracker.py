"""Main concept tracking orchestrator"""

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from backend.models import TrackingRequest, TrackingResponse, TrackingStep, Paper
from backend.arxiv_client import ArXivClient
from backend.embedding_service import EmbeddingService
from backend.kalman_tracker import KalmanConceptTracker
from backend.config import settings

logger = logging.getLogger(__name__)


class ConceptTracker:
    """Main orchestrator for linear concept tracking"""

    def __init__(self):
        """Initialize tracker with required services"""
        logger.info("Initializing ConceptTracker")
        self.arxiv_client = ArXivClient()
        self.embedding_service = EmbeddingService()
        logger.info("ConceptTracker initialized successfully")

    def track(self, request: TrackingRequest) -> TrackingResponse:
        """
        Execute linear concept tracking from seed papers

        Args:
            request: TrackingRequest with seed IDs and parameters

        Returns:
            TrackingResponse with linear timeline

        Raises:
            ValueError: If seed papers cannot be fetched or have no embeddings
        """
        logger.info(f"=" * 80)
        logger.info(f"Starting tracking with {len(request.seed_paper_ids)} seeds")
        logger.info(f"Window: {request.window_months} months, Threshold: {request.similarity_threshold}")
        logger.info(f"=" * 80)

        # 1. Fetch and embed seed papers
        logger.info("Step 1: Fetching seed papers")
        seed_papers = self.arxiv_client.get_papers_by_ids(request.seed_paper_ids)

        if not seed_papers:
            raise ValueError("No seed papers could be fetched")

        logger.info(f"Fetched {len(seed_papers)} seed papers")

        # Embed seed papers
        logger.info("Embedding seed papers")
        for paper in seed_papers:
            embedding = self.embedding_service.embed_paper(paper)
            paper.embedding = embedding.tolist()

        # 2. Initialize Kalman tracker
        logger.info("Step 2: Initializing Kalman tracker")
        tracker = KalmanConceptTracker()
        tracker.initialize(seed_papers)

        # 3. Track through time windows
        logger.info("Step 3: Tracking through time windows")
        timeline = []
        current_date = max(p.published for p in seed_papers)

        # Parse end_date and make it timezone-aware (UTC) to match ArXiv papers
        end_date = datetime.fromisoformat(request.end_date)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        step_number = 0

        logger.info(f"Tracking from {current_date.date()} to {end_date.date()}")

        while current_date < end_date:
            step_number += 1
            window_end = current_date + timedelta(days=30 * request.window_months)

            if window_end > end_date:
                window_end = end_date

            logger.info(f"\n{'='*60}")
            logger.info(f"STEP {step_number}: {current_date.date()} to {window_end.date()}")
            logger.info(f"{'='*60}")

            # Get candidate papers in window
            logger.info("Fetching candidate papers from ArXiv")
            candidates = self.arxiv_client.search_papers(
                query="cat:cs.LG OR cat:cs.CL OR cat:cs.AI",
                start_date=current_date,
                end_date=window_end,
                max_results=request.max_papers_per_window
            )

            if not candidates:
                logger.warning(f"No candidates found in window. Stopping.")
                break

            logger.info(f"Found {len(candidates)} candidates")

            # Embed candidates
            logger.info("Embedding candidate papers")
            for paper in candidates:
                embedding = self.embedding_service.embed_paper(paper)
                paper.embedding = embedding.tolist()

            # Evaluate with Kalman filter
            logger.info("Evaluating candidates with Kalman filter")
            accepted, rejected = tracker.process_papers(candidates)

            if not accepted:
                logger.warning(f"No papers accepted in step {step_number}. Concept tracking stopped.")
                break

            # Update tracker state
            logger.info("Updating tracker state")
            tracker.update(accepted)

            # Create timeline step
            accepted_papers = [paper for paper, sim, conf, _ in accepted]
            similarities = [sim for _, sim, _, _ in accepted]
            confidences = [conf for _, _, conf, _ in accepted]

            # Annotate papers with similarity scores
            for paper, sim in zip(accepted_papers, similarities):
                paper.similarity = sim

            # Count confidence tiers
            num_high = sum(1 for c in confidences if c > 0.85)
            num_moderate = sum(1 for c in confidences if 0.75 <= c <= 0.85)
            num_low = sum(1 for c in confidences if c < 0.75)

            step = TrackingStep(
                step_number=step_number,
                start_date=current_date,
                end_date=window_end,
                papers=accepted_papers,
                concept_vector=tracker.position.tolist(),
                velocity=tracker.velocity.tolist(),
                avg_similarity=float(np.mean(similarities)),
                num_high_confidence=num_high,
                num_moderate=num_moderate,
                num_low=num_low
            )

            timeline.append(step)

            logger.info(f"Step complete: {len(accepted_papers)} papers accepted")
            logger.info(f"  Avg similarity: {step.avg_similarity:.3f}")
            logger.info(f"  Confidence tiers: High={num_high}, Moderate={num_moderate}, Low={num_low}")

            # Move to next window
            current_date = window_end

        # 4. Build response
        total_papers = sum(len(step.papers) for step in timeline)

        response = TrackingResponse(
            seed_papers=seed_papers,
            timeline=timeline,
            total_papers=total_papers,
            num_steps=len(timeline),
            date_range=(
                seed_papers[0].published.isoformat(),
                timeline[-1].end_date.isoformat() if timeline else end_date.isoformat()
            ),
            window_months=request.window_months,
            similarity_threshold=request.similarity_threshold
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"TRACKING COMPLETE")
        logger.info(f"  Total steps: {len(timeline)}")
        logger.info(f"  Total papers: {total_papers}")
        logger.info(f"  Date range: {response.date_range[0]} to {response.date_range[1]}")
        logger.info(f"{'='*80}\n")

        return response
