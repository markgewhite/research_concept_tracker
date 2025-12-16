---
title: ArXiv Concept Tracker
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.10.0
app_file: app.py
pinned: false
---

# ArXiv Concept Tracker

Track how research concepts evolve over time using AI-powered semantic embeddings and Kalman filtering.

## Features

- 🔍 **Search ArXiv papers** by keyword with optional year filtering
- 📊 **Track concept evolution** through time windows
- 🧠 **Semantic embeddings** using sentence-transformers (MiniLM-L6-v2)
- 📈 **Interactive timeline** visualization with Gradio
- 🎯 **Kalman filter** constraints prevent unrealistic concept jumps
- 💾 **Local embeddings cache** - no API costs
- 🌐 **Works locally and on HuggingFace Spaces**

## Quick Start

### Local Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd concept_tracker
```

2. **Create and activate virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- Gradio (web interface)
- sentence-transformers (embeddings)
- ArXiv API client
- NumPy, scikit-learn (computations)
- Pytest (testing)

**Note**: First run will download the embedding model (~400MB) automatically.

### Run Locally

```bash
python app.py
```

This will:
- Start a local server at `http://127.0.0.1:7860`
- Open the interface in your browser
- Generate embeddings cache at `cache/embeddings/`

### Get a Public URL (for showcasing)

```bash
python app.py --share
```

This generates a temporary public URL (valid for 72 hours) that you can share:
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://abc123xyz.gradio.live  ← Share this!
```

### Deploy to HuggingFace Spaces (permanent hosting)

1. **Create a Space** on [HuggingFace Spaces](https://huggingface.co/spaces)
2. **Push your code**:
```bash
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/concept-tracker
git push hf main
```

Your app will be live at: `https://huggingface.co/spaces/YOUR-USERNAME/concept-tracker`

## How to Use

### 1. Search for Seed Papers

- Enter a search query (e.g., "attention is all you need", "diffusion models")
- Optionally filter by year range
- Select 1-5 papers that define your concept

### 2. Configure Tracking

- **End Date**: How far forward to track (auto-calculated as seed date + 2 years)
- **Window Size**: Time window for each step (default: 6 months)
- **Max Papers**: Papers to fetch per window (500-2000 for GPU, 50-100 for CPU)

### 3. View Results

- Timeline showing concept evolution across time windows
- Papers ranked by similarity to concept
- Statistics: high/moderate/low confidence counts

## How It Works

### Concept Tracking Algorithm

1. **Initialization**: Start with 1-5 seed papers that define your concept
2. **Embedding**: Generate semantic embeddings (title + abstract) using sentence-transformers
3. **Time Windows**: Move forward in configurable windows (default: 6 months)
4. **For each window**:
   - Fetch candidate papers from ArXiv (cs.LG, cs.CL, cs.AI)
   - Generate embeddings (cached after first generation)
   - **Kalman Filtering**: Evaluate each paper against physics-inspired constraints:
     - **Similarity**: Must be ≥ 0.50 to current concept vector
     - **Velocity**: Concept drift must be < 0.05 (prevents sudden jumps)
     - **Acceleration**: Change in velocity must be < 0.02 (prevents direction shifts)
   - Accept papers that pass all constraints
   - Update concept vector as weighted mean of accepted papers
5. **Repeat** until end date

### Kalman Filter Validation

The tracker uses physics-inspired constraints to reject papers that would cause unrealistic concept jumps:

- **Similarity < 0.50**: Too dissimilar to current concept
- **Velocity > 0.05**: Concept jumping too fast through embedding space
- **Acceleration > 0.02**: Sudden change in direction

This ensures smooth, realistic concept evolution tracking.

## Configuration

The application uses sensible defaults. To customize, edit `backend/config.py`:

```python
# Kalman Filter Parameters
max_velocity = 1.0        # Max concept drift per time step
max_acceleration = 0.6    # Max change in velocity

# Similarity Thresholds
threshold_auto_include = 0.85  # High confidence (auto-accept)
threshold_strong = 0.75        # Strong confidence
threshold_moderate = 0.60      # Moderate confidence
threshold_reject = 0.50        # Below this = reject
```

### Tuning Tips

**Too strict (rejecting true positives)?**
- Increase `max_velocity` (e.g., 1.5)
- Increase `max_acceleration` (e.g., 0.8)
- Lower `threshold_reject` (e.g., 0.45)

**Too loose (accepting false positives)?**
- Decrease `max_velocity` (e.g., 0.5)
- Decrease `max_acceleration` (e.g., 0.3)
- Raise `threshold_reject` (e.g., 0.60)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_kalman.py -v

# Run integration tests (uses real ArXiv data)
pytest tests/test_arxiv_client.py -v
```

## Project Structure

```
concept_tracker/
├── app.py                      # Gradio interface
├── backend/
│   ├── gradio_wrapper.py       # Gradio event handlers
│   ├── tracker.py              # Main orchestrator
│   ├── arxiv_client.py         # ArXiv API wrapper
│   ├── embedding_service.py    # Embeddings + cache
│   ├── kalman_tracker.py       # Kalman filter logic
│   ├── config.py               # Configuration
│   ├── models.py               # Pydantic data models
│   └── utils/
│       └── cache.py            # Embedding cache
├── cache/                      # Auto-created embedding cache
├── tests/                      # Test suite
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Performance

### First Run
- **Time**: 10-15 minutes
- **Bottleneck**: Model download (~400MB) + initial embedding generation

### Subsequent Runs (Cached)
- **Time**: 2-3 minutes
- **Bottleneck**: ArXiv API queries + Kalman filtering

### Optimizations
- All embeddings permanently cached in `cache/embeddings/`
- Cache grows ~4KB per paper
- 10,000 papers ≈ 40MB cache

### HuggingFace Spaces Performance
- **Free tier**: CPU-only, 16GB RAM, slower but functional
- **ZeroGPU option**: Faster inference (requires configuration)

## Troubleshooting

### Model won't download
**Solution**: Ensure ~1GB free disk space. Model downloads to `~/.cache/huggingface/`

### ArXiv API errors (429, timeouts)
**Solution**: Built-in rate limiting (3 sec delay). If errors persist, increase `arxiv_rate_limit` in config.

### No papers accepted in tracking
**Solution**:
1. Check console logs for rejection reasons
2. Lower `threshold_reject` in config
3. Increase `max_velocity` if velocity rejections are common

### Out of memory
**Solution**: Reduce `max_papers_per_window` (try 50-100 on CPU, 500-2000 on GPU)

## Example: Track Transformer Evolution

1. Search: "attention is all you need"
2. Select the 2017 paper (arxiv:1706.03762)
3. Set end date: 2018-12-31
4. Window: 6 months
5. Max papers: 500
6. Click "Track Concept Evolution"

**Expected results**:
- Should find BERT and other transformer variants
- Should NOT jump to unrelated NLP (pure RNN papers)
- Similarity should stay above 0.50
- 2-3 time steps with 10-30 papers each

## Future Enhancements

- ✅ **Linear tracking** (current implementation)
- 🔲 **Tree branching** with HDBSCAN clustering
- 🔲 **Bidirectional tracking** (trace concepts to origins)
- 🔲 **Multi-signal validation** (citations, author overlap)
- 🔲 **Export results** to JSON/CSV

## License

MIT License

## Acknowledgments

- ArXiv for open access to research papers
- HuggingFace for sentence-transformers and Spaces hosting
- Gradio for the web interface framework
