"""
Gradio-friendly wrappers for backend services

This module provides simplified interfaces for Gradio event handlers,
handling error formatting, progress updates, and state management.
"""

import csv
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import pandas as pd

from backend.arxiv_client import ArXivClient
from backend.tracker import ConceptTracker
from backend.models import Paper, TrackingRequest, TrackingResponse
from backend.visualization import create_tsne_visualization

logger = logging.getLogger(__name__)


class GradioConceptTracker:
    """Gradio-friendly wrapper for ConceptTracker"""

    def __init__(self):
        self.arxiv_client = ArXivClient()
        # Lazy initialization: Don't create ConceptTracker until tracking time
        # This prevents CUDA initialization during search (HuggingFace ZeroGPU requirement)
        self.tracker = None
        # Store last tracking response for export functionality
        self._last_response: Optional[TrackingResponse] = None

    def search_papers(
        self,
        query: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        limit: int = 20
    ) -> Tuple[pd.DataFrame, Dict[str, Paper], str]:
        """
        Search ArXiv and return Gradio-friendly dataframe

        Returns:
            (dataframe, papers_dict, status_message)
        """
        if not query.strip():
            return pd.DataFrame(), {}, "❌ Please enter a search query"

        try:
            # Convert years to datetime
            start_date = datetime(start_year, 1, 1) if start_year else None
            end_date = datetime(end_year, 12, 31) if end_year else None

            # Search ArXiv
            papers = self.arxiv_client.search_papers(
                query, start_date, end_date, limit
            )

            if not papers:
                return pd.DataFrame(), {}, "No papers found"

            # Build papers dict
            papers_dict = {p.arxiv_id: p for p in papers}

            # Convert to dataframe
            rows = []
            for p in papers:
                rows.append({
                    "Title": p.title[:80] + "..." if len(p.title) > 80 else p.title,
                    "Authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
                    "Year": p.published.year,
                    "ArXiv ID": p.arxiv_id
                })

            df = pd.DataFrame(rows)
            status = f"✅ Found {len(papers)} papers"

            return df, papers_dict, status

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return pd.DataFrame(), {}, f"❌ Search failed: {str(e)}"

    def track_concept(
        self,
        seed_ids: List[str],
        end_date_str: str,
        window_months: int,
        max_papers: int,
        progress=None
    ) -> Tuple[str, Dict, str, object]:
        """
        Track concept evolution (Gradio-friendly)

        Returns:
            (timeline_html, results_dict, status_message, visualization_figure)
        """
        if not seed_ids:
            return "", {}, "❌ Please select at least one seed paper", None

        try:
            if progress is not None:
                progress(0, desc="Starting tracker...")

            # Lazy initialization: Create tracker only when needed
            # This ensures model loading happens inside @spaces.GPU decorated function
            if self.tracker is None:
                self.tracker = ConceptTracker()

            # Build tracking request
            request = TrackingRequest(
                seed_paper_ids=seed_ids,
                end_date=end_date_str,
                window_months=window_months,
                similarity_threshold=0.50,
                max_papers_per_window=max_papers
            )

            if progress is not None:
                progress(0.1, desc="Initializing tracking...")

            # Execute tracking with progress callback
            response = self.tracker.track(request, progress_callback=progress)

            if progress is not None:
                progress(1.0, desc="Complete!")

            # Store response for export
            self._last_response = response

            # Format results for Gradio
            timeline_html = self._format_timeline_html(response)
            results_dict = self._format_results_dict(response)
            status = f"✅ Tracked {response.total_papers} papers across {response.num_steps} steps"

            # Generate t-SNE visualization
            if progress is not None:
                progress(0.95, desc="Generating visualization...")

            try:
                viz_figure = create_tsne_visualization(response)
            except Exception as viz_error:
                logger.warning(f"Visualization failed: {viz_error}")
                viz_figure = None

            return timeline_html, results_dict, status, viz_figure

        except Exception as e:
            logger.error(f"Tracking failed: {e}", exc_info=True)
            return "", {}, f"❌ Tracking failed: {str(e)}", None

    def _format_timeline_html(self, response: TrackingResponse) -> str:
        """Convert TrackingResponse to rich HTML for Gradio display"""
        html = "<div style='font-family: sans-serif;'>"

        for step in response.timeline:
            html += f"""
            <div style='border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px;'>
                <h3>Step {step.step_number}: {step.start_date.strftime('%b %Y')} → {step.end_date.strftime('%b %Y')}</h3>
                <p><b>{len(step.papers)} papers</b> | Avg similarity: {step.avg_similarity:.3f} | Position drift: {step.position_drift:.4f}</p>

                <div style='display: flex; gap: 12px; margin: 12px 0;'>
                    <span style='background: #d1fae5; padding: 4px 8px; border-radius: 4px; color: #000;'>
                        High (≥0.70): {step.num_high_confidence}
                    </span>
                    <span style='background: #fef3c7; padding: 4px 8px; border-radius: 4px; color: #000;'>
                        Moderate (0.60-0.70): {step.num_moderate}
                    </span>
                    <span style='background: #fed7aa; padding: 4px 8px; border-radius: 4px; color: #000;'>
                        Low (<0.60): {step.num_low}
                    </span>
                </div>

                <details>
                    <summary style='cursor: pointer; color: #2563eb;'>View papers</summary>
                    <ul style='margin-top: 8px;'>
            """

            # Sort by similarity, show top 10
            top_papers = sorted(step.papers, key=lambda p: p.similarity or 0, reverse=True)[:10]
            for paper in top_papers:
                sim_color = "#16a34a" if paper.similarity >= 0.70 else "#ca8a04" if paper.similarity >= 0.60 else "#ea580c"
                html += f"""
                <li style='margin-bottom: 8px;'>
                    <span style='color: {sim_color}; font-weight: bold;'>{paper.similarity:.3f}</span>
                    - {paper.title}
                    <br><small style='color: #666;'>{paper.arxiv_id} | <a href="{paper.pdf_url}" target="_blank">PDF</a></small>
                </li>
                """

            html += """
                    </ul>
                </details>
            </div>
            """

        html += "</div>"
        return html

    def _format_results_dict(self, response: TrackingResponse) -> Dict:
        """Convert TrackingResponse to dict for stats display"""
        return {
            "num_steps": response.num_steps,
            "total_papers": response.total_papers,
            "avg_similarity": sum(s.avg_similarity for s in response.timeline) / len(response.timeline) if response.timeline else 0,
            "window_months": response.window_months,
            "response": response
        }

    def export_json(self) -> Optional[str]:
        """
        Export last tracking results to JSON file.

        Returns:
            Path to temporary JSON file, or None if no data
        """
        if self._last_response is None:
            return None

        try:
            json_data = self._last_response.model_dump_json(indent=2)

            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                prefix='concept_tracker_',
                delete=False
            )
            temp_file.write(json_data)
            temp_file.close()

            logger.info(f"Exported JSON to {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return None

    def export_csv(self) -> Optional[str]:
        """
        Export last tracking results to CSV file (flattened paper records).

        Returns:
            Path to temporary CSV file, or None if no data
        """
        if self._last_response is None:
            return None

        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.csv',
                prefix='concept_tracker_',
                delete=False,
                newline=''
            )

            fieldnames = [
                'step_number', 'step_start_date', 'step_end_date',
                'step_avg_similarity', 'step_position_drift',
                'arxiv_id', 'title', 'authors', 'published',
                'categories', 'similarity', 'pdf_url'
            ]

            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()

            # Write seed papers (step 0)
            for paper in self._last_response.seed_papers:
                writer.writerow({
                    'step_number': 0,
                    'step_start_date': '',
                    'step_end_date': '',
                    'step_avg_similarity': '',
                    'step_position_drift': '',
                    'arxiv_id': paper.arxiv_id,
                    'title': paper.title,
                    'authors': '; '.join(paper.authors),
                    'published': paper.published.strftime('%Y-%m-%d'),
                    'categories': '; '.join(paper.categories),
                    'similarity': paper.similarity if paper.similarity else '',
                    'pdf_url': paper.pdf_url
                })

            # Write timeline papers
            for step in self._last_response.timeline:
                for paper in step.papers:
                    writer.writerow({
                        'step_number': step.step_number,
                        'step_start_date': step.start_date.strftime('%Y-%m-%d'),
                        'step_end_date': step.end_date.strftime('%Y-%m-%d'),
                        'step_avg_similarity': f'{step.avg_similarity:.4f}',
                        'step_position_drift': f'{step.position_drift:.4f}',
                        'arxiv_id': paper.arxiv_id,
                        'title': paper.title,
                        'authors': '; '.join(paper.authors),
                        'published': paper.published.strftime('%Y-%m-%d'),
                        'categories': '; '.join(paper.categories),
                        'similarity': f'{paper.similarity:.4f}' if paper.similarity else '',
                        'pdf_url': paper.pdf_url
                    })

            temp_file.close()
            logger.info(f"Exported CSV to {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return None
