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
    logger.info("Creating t-SNE visualization")

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

    logger.info(f"Collected {len(all_embeddings)} paper embeddings and {len(concept_vectors)} concept vectors")
    logger.info(f"Embedding shape: {all_embeddings.shape}")
    logger.info(f"Steps range: {all_steps.min()} to {all_steps.max()}")

    # 2. Run t-SNE on all embeddings (papers + concept vectors)
    # Combine for consistent projection
    concept_vectors_array = np.array(concept_vectors)
    combined = np.vstack([all_embeddings, concept_vectors_array])

    logger.info(f"Running t-SNE on {combined.shape[0]} points in {combined.shape[1]}D space")

    # Ensure perplexity is valid (must be < n_samples)
    perplexity = min(30, max(5, len(combined) - 1))
    logger.info(f"Using perplexity: {perplexity}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embedded = tsne.fit_transform(combined)

    logger.info(f"t-SNE output shape: {embedded.shape}")
    logger.info(f"t-SNE output range: x=[{embedded[:, 0].min():.2f}, {embedded[:, 0].max():.2f}], y=[{embedded[:, 1].min():.2f}, {embedded[:, 1].max():.2f}]")

    # Split back into papers and concept trajectory
    paper_coords = embedded[:len(all_embeddings)]
    concept_coords = embedded[len(all_embeddings):]

    logger.info("t-SNE projection complete")

    # 3. Create color scale (red to blue spectrum)
    num_steps = len(response.timeline) + 1  # +1 for seed papers
    colors = []
    for step in all_steps:
        # Interpolate from red (step 0) to blue (final step)
        ratio = step / max(num_steps - 1, 1)
        r = int(255 * (1 - ratio))
        b = int(255 * ratio)
        g = 0
        colors.append(f'rgb({r},{g},{b})')

    # 4. Create plotly figure
    fig = go.Figure()

    # Add papers as scatter points
    hover_texts = []
    for i, paper in enumerate(all_papers):
        step = all_steps[i]
        step_label = "Seed" if step == 0 else f"Step {step}"
        sim_text = f"Similarity: {paper.similarity:.3f}<br>" if hasattr(paper, 'similarity') and paper.similarity is not None else ""
        hover_text = (
            f"<b>{paper.title[:60]}...</b><br>"
            f"{step_label}<br>"
            f"Year: {paper.published.year}<br>"
            f"ArXiv: {paper.arxiv_id}<br>"
            f"{sim_text}"
        )
        hover_texts.append(hover_text)

    logger.info(f"Creating scatter plot with {len(paper_coords)} points")
    logger.info(f"X coords range: [{paper_coords[:, 0].min():.2f}, {paper_coords[:, 0].max():.2f}]")
    logger.info(f"Y coords range: [{paper_coords[:, 1].min():.2f}, {paper_coords[:, 1].max():.2f}]")

    fig.add_trace(go.Scatter(
        x=paper_coords[:, 0],
        y=paper_coords[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,
            line=dict(width=1, color='white'),
            opacity=0.7
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Papers'
    ))

    # Add concept trajectory as line
    fig.add_trace(go.Scatter(
        x=concept_coords[:, 0],
        y=concept_coords[:, 1],
        mode='lines+markers',
        line=dict(color='black', width=3, dash='solid'),
        marker=dict(size=12, color='yellow', symbol='star', line=dict(width=2, color='black')),
        text=[f"Step {i+1}" for i in range(len(concept_coords))],
        hovertemplate='Concept Position<br>Step %{text}<extra></extra>',
        name='Concept Trajectory'
    ))

    # 5. Layout
    fig.update_layout(
        title={
            'text': 'Concept Evolution Visualization (t-SNE)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        hovermode='closest',
        width=900,
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        plot_bgcolor='#f8f9fa',
        font=dict(family='Arial, sans-serif', size=12)
    )

    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    logger.info("Visualization created successfully")
    return fig
