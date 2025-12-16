"""Gradio application for ArXiv Concept Tracker"""

import gradio as gr
from backend.gradio_wrapper import GradioConceptTracker

# Initialize tracker
tracker = GradioConceptTracker()

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.tab-nav button {
    font-size: 16px;
    font-weight: 500;
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
                    label="Start Year (optional)",
                    precision=0,
                    minimum=1990,
                    maximum=2025
                )
            with gr.Column(scale=1):
                end_year = gr.Number(
                    label="End Year (optional)",
                    precision=0,
                    minimum=1990,
                    maximum=2025
                )

        with gr.Row():
            search_btn = gr.Button("🔍 Search ArXiv", variant="primary", size="lg")
            clear_year_btn = gr.Button("Clear Year Filter", size="sm")

        search_status = gr.Textbox(label="Status", interactive=False)

        search_results = gr.Dataframe(
            headers=["Select", "Title", "Authors", "Year", "ArXiv ID"],
            datatype=["bool", "str", "str", "number", "str"],
            col_count=(5, "fixed"),
            interactive=True,
            label="Search Results (click checkbox to select as seed)"
        )

    # Tab 2: Configure tracking
    with gr.Tab("2. Configure Tracking"):
        gr.Markdown("### Configure tracking parameters")

        selected_display = gr.Dataframe(
            headers=["Title", "Authors", "Year", "ArXiv ID"],
            datatype=["str", "str", "number", "str"],
            label="Selected Seed Papers",
            interactive=False
        )

        with gr.Row():
            with gr.Column():
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    placeholder="Auto-calculated based on seeds",
                    info="Tracker will follow concept evolution until this date"
                )
            with gr.Column():
                window_months = gr.Slider(
                    minimum=1,
                    maximum=24,
                    value=6,
                    step=1,
                    label="Window Size (months)",
                    info="Time window for each tracking step"
                )

        max_papers = gr.Slider(
            minimum=50,
            maximum=2000,
            value=500,
            step=50,
            label="Max Papers per Window",
            info="GPU: 500-2000 recommended. CPU: 50-100 max."
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

        timeline_display = gr.HTML(
            label="Timeline",
            value="<p style='color: #666;'>Run tracking to see results...</p>"
        )

        with gr.Row():
            export_json_btn = gr.Button("📥 Export JSON")
            export_csv_btn = gr.Button("📥 Export CSV")

if __name__ == "__main__":
    app.launch()
