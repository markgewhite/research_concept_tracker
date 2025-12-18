"""Visualization utilities for concept tracking results"""

import numpy as np
import logging
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend.models import TrackingResponse

logger = logging.getLogger(__name__)


def create_tsne_visualization(response: TrackingResponse) -> go.Figure:
    """
    Create interactive t-SNE visualization of concept evolution

    Args:
        response: TrackingResponse with timeline and embeddings

    Returns:
        Plotly figure with interactive visualization
    """
    # 1. Collect all embeddings and metadata
    all_embeddings = []
    all_papers = []
    all_steps = []
    concept_vectors = []

    # Add seed papers (step 0)
    for paper in response.seed_papers:
        if paper.embedding is None:
            logger.warning(f"Seed paper {paper.arxiv_id} has no embedding, skipping")
            continue
        all_embeddings.append(paper.embedding)
        all_papers.append(paper)
        all_steps.append(0)

    # Add papers from each step
    for step in response.timeline:
        concept_vectors.append(step.concept_vector)
        for paper in step.papers:
            if paper.embedding is None:
                logger.warning(f"Paper {paper.arxiv_id} in step {step.step_number} has no embedding, skipping")
                continue
            all_embeddings.append(paper.embedding)
            all_papers.append(paper)
            all_steps.append(step.step_number)

    if len(all_embeddings) == 0:
        raise ValueError("No papers with embeddings found!")

    all_embeddings = np.array(all_embeddings)
    all_steps = np.array(all_steps)

    # 2. Run t-SNE on all embeddings (papers + concept vectors)
    # Combine for consistent projection
    concept_vectors_array = np.array(concept_vectors)
    combined = np.vstack([all_embeddings, concept_vectors_array])

    # Ensure perplexity is valid (must be < n_samples)
    perplexity = min(40, max(5, len(combined) - 1))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedded = tsne.fit_transform(combined)

    # Split back into papers and concept trajectory
    paper_coords = embedded[:len(all_embeddings)]
    concept_coords = embedded[len(all_embeddings):]

    # 3. Create color mapping for steps (red to blue spectrum)
    num_steps = len(response.timeline) + 1  # +1 for seed papers

    def get_step_color(step):
        """Get RGB color for a given step"""
        ratio = step / max(num_steps - 1, 1)
        r = int(255 * (1 - ratio))
        b = int(255 * ratio)
        g = 0
        return f'rgb({r},{g},{b})'

    # 4. Create plotly figure
    fig = go.Figure()

    # Use Scattergl for better rendering
    from plotly.graph_objs import Scattergl

    # Add papers grouped by step (for proper legend)
    for step_num in range(num_steps):
        # Get papers for this step
        step_mask = all_steps == step_num
        step_indices = np.where(step_mask)[0]

        if len(step_indices) == 0:
            continue

        # Get coordinates for this step
        step_x = paper_coords[step_indices, 0].tolist()
        step_y = paper_coords[step_indices, 1].tolist()

        # Create hover text for this step
        step_hover_texts = []
        for idx in step_indices:
            paper = all_papers[idx]
            step_label = "Seed Papers" if step_num == 0 else f"Step {step_num}"
            sim_text = f"Similarity: {paper.similarity:.3f}<br>" if hasattr(paper, 'similarity') and paper.similarity is not None else ""
            hover_text = (
                f"<b>{paper.title[:60]}...</b><br>"
                f"{step_label}<br>"
                f"Year: {paper.published.year}<br>"
                f"ArXiv: {paper.arxiv_id}<br>"
                f"{sim_text}"
            )
            step_hover_texts.append(hover_text)

        # Determine label and color
        if step_num == 0:
            label = "Seed Papers"
            color = 'rgb(0,0,0)'  # Black for seed papers
        else:
            # Get date range for this step
            step_data = response.timeline[step_num - 1]
            label = step_data.start_date.strftime('%b %Y')
            color = get_step_color(step_num)

        # Add trace for this step
        fig.add_trace(Scattergl(
            x=step_x,
            y=step_y,
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                line=dict(width=0.3, color='rgba(255,255,255,0.5)'),
                opacity=0.7
            ),
            text=step_hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name=label,
            legendgroup=f'step{step_num}',
            showlegend=True
        ))

    # Add concept trajectory as line with arrow
    fig.add_trace(go.Scatter(
        x=concept_coords[:, 0],
        y=concept_coords[:, 1],
        mode='lines',
        line=dict(color='black', width=2),
        text=[f"Step {i+1}" for i in range(len(concept_coords))],
        hovertemplate='Concept Trajectory<br>Step %{text}<extra></extra>',
        name='Concept Trajectory',
        showlegend=True
    ))

    # Add arrow at the end of trajectory
    if len(concept_coords) >= 2:
        # Get last two points for arrow direction
        x_end = concept_coords[-1, 0]
        y_end = concept_coords[-1, 1]
        x_prev = concept_coords[-2, 0]
        y_prev = concept_coords[-2, 1]

        fig.add_annotation(
            x=x_end,
            y=y_end,
            ax=x_prev,
            ay=y_prev,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='black'
        )

    # 5. Layout with explicit ranges
    fig.update_layout(
        title={
            'text': f'Concept Evolution Visualization (t-SNE, perplexity={perplexity})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis=dict(
            title='t-SNE Dimension 1',
            range=[paper_coords[:, 0].min() - 1, paper_coords[:, 0].max() + 1],
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='t-SNE Dimension 2',
            range=[paper_coords[:, 1].min() - 1, paper_coords[:, 1].max() + 1],
            showgrid=True,
            gridcolor='lightgray'
        ),
        hovermode='closest',
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=80, r=150, t=80, b=80)  # Right margin for legend
    )

    # Configure for Gradio display
    fig.update_layout(
        autosize=True,
        template='plotly_white'
    )

    return fig
