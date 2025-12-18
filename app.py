"""Gradio application for ArXiv Concept Tracker"""

import logging
import os
import gradio as gr
from backend.gradio_wrapper import GradioConceptTracker

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.info(f"Starting ArXiv Concept Tracker with log level: {log_level}")

# Lazy initialization - create tracker on first use
_tracker = None

def get_tracker():
    """Get or create tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = GradioConceptTracker()
    return _tracker

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.tab-nav button {
    font-size: 16px;
    font-weight: 500;
}
/* Force checkboxes to display vertically, one per line */
#seed_selection label {
    display: block !important;
    width: 100% !important;
    margin-bottom: 8px !important;
}
/* Fix selected seeds table display */
#selected_seeds_table {
    min-height: 100px;
    max-height: 200px;
    overflow-y: auto;
}
#selected_seeds_table .table-wrap {
    overflow: visible !important;
}
/* Align tracking params row - equal height columns */
#tracking_params_row {
    align-items: stretch !important;
}
#tracking_params_row > div {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-end !important;
}
"""

with gr.Blocks(title="ArXiv Concept Tracker", css=custom_css) as app:
    gr.Markdown("""
    # 📚 ArXiv Concept Tracker

    Track how research concepts evolve over time using semantic embeddings and Kalman filtering.

    **How it works:**
    1. Search for seed papers that define your concept
    2. Configure tracking parameters (time range, window size)
    3. View timeline of concept evolution with related papers
    """)

    # State management
    selected_seeds = gr.State(value=[])
    seed_papers_data = gr.State(value={})

    # Tab 1: Search for seed papers
    with gr.Tab("1. Find Seed Papers"):
        gr.Markdown("### Search ArXiv for papers that define your concept")
        gr.Markdown("Select up to 5 seed papers to initialize the tracker.")

        with gr.Row():
            with gr.Column(scale=3):
                search_query = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., 'attention mechanism', 'transformer', 'BERT'",
                    lines=1
                )
            with gr.Column(scale=1):
                start_year = gr.Number(
                    label="Start Year",
                    value=2015,
                    precision=0,
                    minimum=1990,
                    maximum=2025
                )
            with gr.Column(scale=1):
                end_year = gr.Number(
                    label="End Year",
                    value=2020,
                    precision=0,
                    minimum=1990,
                    maximum=2025
                )

        with gr.Row():
            search_btn = gr.Button("🔍 Search ArXiv", variant="primary", size="lg")
            clear_year_btn = gr.Button("Reset Years to Default", size="sm")

        search_status = gr.Textbox(label="Status", interactive=False)

        seed_selection = gr.CheckboxGroup(
            label="Select Seed Papers (max 5)",
            choices=[],
            interactive=True,
            info="Choose papers that define your concept",
            elem_id="seed_selection"
        )

    # Tab 2: Configure tracking
    with gr.Tab("2. Configure Tracking"):
        gr.Markdown("### Configure tracking parameters")

        selected_display = gr.Dataframe(
            headers=["Title", "Authors", "Year", "ArXiv ID"],
            datatype=["str", "str", "number", "str"],
            label="Selected Seed Papers",
            interactive=False,
            wrap=True,
            elem_id="selected_seeds_table"
        )

        with gr.Row(elem_id="tracking_params_row"):
            with gr.Column(scale=2):
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    placeholder="Auto-calculated based on seeds",
                    info="Tracker will follow concept evolution until this date"
                )
            with gr.Column(scale=2):
                window_months = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=3,
                    step=1,
                    label="Window Size (months)",
                    info="Time window for each tracking step"
                )
            with gr.Column(scale=3):
                max_papers = gr.Slider(
                    minimum=50,
                    maximum=2000,
                    value=250,
                    step=50,
                    label="Max Papers per Window",
                    info="Recommended: 250 for 3-month windows"
                )

        track_btn = gr.Button("🚀 Track Concept Evolution", variant="primary", size="lg")
        track_status = gr.Textbox(label="Status", interactive=False)

    # Tab 3: Results
    with gr.Tab("3. Results"):
        gr.Markdown("### Concept Evolution Timeline")

        with gr.Row():
            stats_total = gr.Number(label="Total Papers", precision=0)
            stats_steps = gr.Number(label="Time Steps", precision=0)
            stats_avg_sim = gr.Number(label="Avg Similarity", precision=3)

        with gr.Tab("Timeline"):
            timeline_display = gr.HTML(
                label="Timeline",
                value="<p style='color: #666;'>Run tracking to see results...</p>"
            )

        with gr.Tab("Visualization"):
            visualization_plot = gr.Plot(
                label="t-SNE Concept Evolution",
                value=None
            )
            gr.Markdown("""
            **How to read this visualization:**
            - Each dot represents a paper, colored by time step (black = seed papers, red to blue = progression)
            - Black line with arrow shows the concept trajectory evolution
            - Seed papers (black) are larger for visibility
            - Legend (bottom right) shows each time step with its date
            - Hover over dots to see paper details
            - Click legend items to show/hide specific time periods
            - Click and drag to pan, scroll to zoom
            """)

        with gr.Row():
            export_json_btn = gr.Button("📥 Export JSON")
            export_csv_btn = gr.Button("📥 Export CSV")

    # Event handlers
    def handle_search(query, start_year, end_year):
        """Search ArXiv and return results"""
        if not query:
            return gr.update(choices=[]), {}, "❌ Please enter a search query"

        try:
            df, papers_dict, status = get_tracker().search_papers(
                query=query,
                start_year=int(start_year) if start_year else None,
                end_year=int(end_year) if end_year else None,
                limit=20
            )

            # Create checkbox choices with format: "{title} ({year}). {authors} [arxiv_id]"
            if papers_dict:
                choices = []
                for arxiv_id, paper in papers_dict.items():
                    # Format authors
                    authors = ", ".join(paper.authors[:3])
                    if len(paper.authors) > 3:
                        authors += " et al."
                    
                    # Format: title (year). authors [arxiv_id]
                    choice = f"{paper.title} ({paper.published.year}). {authors} [{arxiv_id}]"
                    choices.append(choice)
            else:
                choices = []

            return gr.update(choices=choices, value=[]), papers_dict, status
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return gr.update(choices=[]), {}, error_msg

    def clear_years():
        """Reset year filters to defaults"""
        return 2015, 2020

    def handle_seed_selection(selected_choices, current_papers):
        """Update selected seeds based on checkbox selection"""
        if not selected_choices:
            return [], current_papers, gr.update(value=[]), "", ""

        # Extract ArXiv IDs from choice strings (format: "Title (year). Authors [arxiv_id]")
        import re
        selected_ids = []
        selected_rows = []

        for choice in selected_choices:
            # Extract arxiv_id from "[arxiv_id]" at end of string
            match = re.search(r'\[([^\]]+)\]$', choice)
            if match:
                arxiv_id = match.group(1)
                if arxiv_id in current_papers:
                    paper = current_papers[arxiv_id]
                    selected_ids.append(arxiv_id)
                    # DataFrame expects list of lists, not list of dicts
                    selected_rows.append([
                        paper.title,
                        ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else ""),
                        paper.published.year,
                        arxiv_id
                    ])

        # Validate max 5 seeds
        if len(selected_ids) > 5:
            # Keep only first 5
            selected_ids = selected_ids[:5]
            selected_rows = selected_rows[:5]
            status = "❌ Maximum 5 seed papers allowed. Keeping first 5 selected."
        else:
            status = f"✅ Selected {len(selected_ids)} seed paper(s)" if selected_ids else ""

        # Auto-calculate end date if seeds selected
        end_date_value = ""
        if selected_ids and current_papers:
            from datetime import datetime, timedelta
            latest_date = None
            for arxiv_id in selected_ids:
                if arxiv_id in current_papers:
                    paper = current_papers[arxiv_id]
                    paper_date = paper.published
                    if latest_date is None or paper_date > latest_date:
                        latest_date = paper_date

            if latest_date:
                end_date = latest_date + timedelta(days=730)  # +2 years
                end_date_value = end_date.strftime("%Y-%m-%d")

        return (
            selected_ids,
            current_papers,
            gr.update(value=selected_rows),
            end_date_value,
            status
        )

    def handle_track(seeds, papers_dict, end_date_str, window_months, max_papers, progress=gr.Progress()):
        """Track concept evolution"""
        if not seeds:
            return "", 0, 0, 0.0, "❌ Please select at least one seed paper", None

        try:
            progress(0, desc="Initializing tracker...")

            timeline_html, results_dict, status, viz_figure = get_tracker().track_concept(
                seed_ids=seeds,
                end_date_str=end_date_str,
                window_months=int(window_months),
                max_papers=int(max_papers),
                progress=progress
            )

            # Extract stats
            total_papers = results_dict.get("total_papers", 0)
            num_steps = results_dict.get("num_steps", 0)
            avg_similarity = results_dict.get("avg_similarity", 0.0)

            return timeline_html, total_papers, num_steps, avg_similarity, status, viz_figure
        except Exception as e:
            import traceback
            error_msg = f"❌ Tracking failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"<p style='color: red;'>{error_msg}</p>", 0, 0, 0.0, error_msg, None

    # Wire up events
    search_btn.click(
        fn=handle_search,
        inputs=[search_query, start_year, end_year],
        outputs=[seed_selection, seed_papers_data, search_status]
    )

    clear_year_btn.click(
        fn=clear_years,
        outputs=[start_year, end_year]
    )

    seed_selection.change(
        fn=handle_seed_selection,
        inputs=[seed_selection, seed_papers_data],
        outputs=[selected_seeds, seed_papers_data, selected_display, end_date, track_status]
    )

    track_btn.click(
        fn=handle_track,
        inputs=[selected_seeds, seed_papers_data, end_date, window_months, max_papers],
        outputs=[timeline_display, stats_total, stats_steps, stats_avg_sim, track_status, visualization_plot]
    )

if __name__ == "__main__":
    app.launch()
