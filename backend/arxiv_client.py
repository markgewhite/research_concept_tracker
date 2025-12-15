"""ArXiv API client wrapper"""

import arxiv
import time
import logging
from datetime import datetime
from typing import Optional
from backend.models import Paper
from backend.config import settings

logger = logging.getLogger(__name__)


class ArXivClient:
    """Wrapper for ArXiv API with rate limiting and error handling"""

    def __init__(self):
        """Initialize ArXiv client with rate limiting"""
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=settings.arxiv_rate_limit,
            num_retries=settings.arxiv_max_retries
        )
        logger.info(f"ArXiv client initialized (rate_limit={settings.arxiv_rate_limit}s, retries={settings.arxiv_max_retries})")

    def get_paper_by_id(self, arxiv_id: str) -> Paper:
        """
        Fetch single paper by ArXiv ID

        Args:
            arxiv_id: ArXiv ID (e.g., "1706.03762")

        Returns:
            Paper object with metadata

        Raises:
            ValueError: If paper not found
        """
        logger.info(f"Fetching paper: {arxiv_id}")

        search = arxiv.Search(id_list=[arxiv_id])
        result = next(self.client.results(search), None)

        if result is None:
            raise ValueError(f"Paper {arxiv_id} not found")

        paper = self._convert_to_paper(result)
        logger.debug(f"Fetched: {paper.title[:50]}...")
        return paper

    def get_papers_by_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        """
        Batch fetch papers by ArXiv IDs

        Args:
            arxiv_ids: List of ArXiv IDs

        Returns:
            List of Paper objects
        """
        logger.info(f"Fetching {len(arxiv_ids)} papers by ID")

        papers = []
        for arxiv_id in arxiv_ids:
            try:
                paper = self.get_paper_by_id(arxiv_id)
                papers.append(paper)
                time.sleep(settings.arxiv_rate_limit)
            except ValueError as e:
                logger.warning(f"Skipping {arxiv_id}: {e}")
                continue

        logger.info(f"Successfully fetched {len(papers)}/{len(arxiv_ids)} papers")
        return papers

    def search_papers(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 100
    ) -> list[Paper]:
        """
        Search ArXiv with optional date filtering

        Args:
            query: Search query (supports ArXiv query syntax)
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects matching query
        """
        # Build query with date filter if provided
        if start_date and end_date:
            date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}2359]"
            full_query = f"({query}) AND {date_query}"
            logger.info(f"Searching: '{query}' from {start_date.date()} to {end_date.date()}")
        else:
            full_query = query
            logger.info(f"Searching: '{query}'")

        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        papers = []
        for result in self.client.results(search):
            papers.append(self._convert_to_paper(result))

        logger.info(f"Found {len(papers)} papers")
        return papers

    def _convert_to_paper(self, result: arxiv.Result) -> Paper:
        """
        Convert arxiv.Result to Paper model

        Args:
            result: ArXiv search result

        Returns:
            Paper object
        """
        # Extract ArXiv ID from entry_id URL
        arxiv_id = result.entry_id.split('/')[-1]

        return Paper(
            arxiv_id=arxiv_id,
            title=result.title,
            abstract=result.summary,
            authors=[author.name for author in result.authors],
            published=result.published,
            updated=result.updated,
            categories=result.categories,
            pdf_url=result.pdf_url
        )
