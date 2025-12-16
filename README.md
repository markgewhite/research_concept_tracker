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

- 🔍 **Search ArXiv papers** by keyword
- 📊 **Track concept evolution** through time windows
- 🧠 **Semantic similarity** with embeddings (MiniLM-L6-v2)
- 📈 **Interactive timeline** visualization
- 🎯 **Kalman filter** for smooth concept tracking
- **Linear concept tracking**: Follow concept evolution from seed papers forward through time
- **Local embeddings**: sentence-transformers (no API costs)
- **Kalman filtering**: Velocity and acceleration constraints prevent unrealistic concept jumps
- **ArXiv integration**: Automatic paper fetching and metadata extraction
- **REST API**: FastAPI backend with JSON responses
- **Comprehensive caching**: Embeddings are cached locally for fast repeated runs

## Quick Start

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd /Users/markgewhite/Documents/MyFiles/Projects/training/ztm/llm_web_apps/concept_tracker
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
- FastAPI & Uvicorn (web framework)
- Qwen3 embeddings via sentence-transformers
- ArXiv API client
- NumPy, scikit-learn for computations
- Pytest for testing

**Note**: First run will download the Qwen3 model (~400MB) automatically.

### Configuration

The application uses sensible defaults. To customize, copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key parameters in `backend/config.py`:

```python
# Kalman Filter Parameters
max_velocity = 0.05       # Max concept drift per time step
max_acceleration = 0.02   # Max change in velocity

# Similarity Thresholds
threshold_auto_include = 0.85  # High confidence (auto-accept)
threshold_strong = 0.75        # Moderate confidence
threshold_moderate = 0.65      # Low confidence (minimum)
```

## Usage

### Start the Server

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Search Papers

Find potential seed papers:

```bash
curl "http://localhost:8000/api/search?query=attention%20is%20all%20you%20need&limit=5"
```

#### 2. Get Single Paper

Get details for a specific paper:

```bash
curl "http://localhost:8000/api/paper/1706.03762"
```

#### 3. Track Concept Evolution

Track a concept from seed papers forward:

```bash
curl -X POST "http://localhost:8000/api/track" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_paper_ids": ["1706.03762"],
    "end_date": "2018-12-31",
    "window_months": 6,
    "max_papers_per_window": 50
  }'
```

**Parameters**:
- `seed_paper_ids`: 1-5 ArXiv IDs to start tracking from
- `end_date`: End date (ISO format: "YYYY-MM-DD")
- `window_months`: Time window size (default: 6 months)
- `max_papers_per_window`: Max papers to fetch per window (default: 50)

### Example: Track Transformer Evolution

```bash
# Track from "Attention is All You Need" (2017) to end of 2018
curl -X POST "http://localhost:8000/api/track" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_paper_ids": ["1706.03762"],
    "end_date": "2018-12-31",
    "window_months": 6,
    "similarity_threshold": 0.65,
    "max_papers_per_window": 50
  }' | python -m json.tool
```

**Expected output**:
```json
{
  "seed_papers": [...],
  "timeline": [
    {
      "step_number": 1,
      "start_date": "2017-06-12T...",
      "end_date": "2017-12-12T...",
      "papers": [...],
      "avg_similarity": 0.78,
      "num_high_confidence": 12,
      "num_moderate": 8,
      "num_low": 3
    },
    ...
  ],
  "total_papers": 45,
  "num_steps": 3
}
```

## How It Works

### Concept Tracking Algorithm

1. **Initialization**: Start with 1-5 seed papers (e.g., "Attention is All You Need")
2. **Embedding**: Generate semantic embeddings (title + abstract) using Qwen3
3. **Time Windows**: Move forward in configurable windows (default: 6 months)
4. **For each window**:
   - Fetch candidate papers from ArXiv
   - Generate embeddings (cached after first generation)
   - **Kalman Filtering**: Evaluate each paper against physics-inspired constraints:
     - **Similarity**: Must be > 0.65 to current concept vector
     - **Velocity**: Change must be < 0.05 (prevents sudden jumps)
     - **Acceleration**: Change in velocity must be < 0.02 (prevents direction shifts)
   - Accept papers that pass all constraints
   - Update concept vector as weighted mean of accepted papers
5. **Repeat** until end date

### Kalman Filter Validation

The tracker rejects papers that would cause unrealistic concept jumps:

- **Similarity < 0.65**: Too dissimilar to current concept
- **Velocity > 0.05**: Concept jumping too fast through embedding space
- **Acceleration > 0.02**: Sudden change in direction

Check logs for rejection reasons:
```bash
uvicorn backend.main:app --log-level=debug
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_kalman.py -v

# Run slow integration tests (uses real ArXiv data)
pytest tests/test_api.py -v -s --tb=short
```

### Test Coverage

- `test_arxiv_client.py`: ArXiv API integration
- `test_kalman.py`: Kalman filter constraints
- `test_api.py`: FastAPI endpoints

## Project Structure

```
concept_tracker/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app & endpoints
│   ├── config.py            # Kalman parameters & settings
│   ├── models.py            # Pydantic data models
│   ├── arxiv_client.py      # ArXiv API wrapper
│   ├── embedding_service.py # Qwen3 embeddings + cache
│   ├── kalman_tracker.py    # Core tracking algorithm
│   ├── tracker.py           # Main orchestrator
│   └── utils/
│       ├── __init__.py
│       └── cache.py         # Pickle-based cache
├── cache/                   # Embedding storage (auto-created)
├── tests/                   # Test suite
├── requirements.txt         # Python dependencies
├── .env.example            # Configuration template
└── README.md               # This file
```

## Performance

### First Run
- **Time**: 10-15 minutes (one-time embedding generation + download)
- **Bottleneck**: Qwen3 model download (~400MB) and embedding generation

### Subsequent Runs (Cached)
- **Time**: 2-3 minutes
- **Bottleneck**: ArXiv API queries and Kalman filtering

### Optimizations
- All embeddings are permanently cached in `cache/embeddings/`
- Cache grows ~4KB per paper (1024 floats × 4 bytes)
- 10,000 papers = ~40MB cache (acceptable)

## Tuning Kalman Parameters

If tracking results are not satisfactory:

### Too Strict (Rejecting True Positives)

Edit `backend/config.py`:
```python
max_velocity = 0.07       # Increase from 0.05
max_acceleration = 0.03   # Increase from 0.02
threshold_moderate = 0.60 # Decrease from 0.65
```

### Too Loose (Accepting False Positives)

Edit `backend/config.py`:
```python
max_velocity = 0.03       # Decrease from 0.05
max_acceleration = 0.01   # Decrease from 0.02
threshold_moderate = 0.70 # Increase from 0.65
```

Restart the server after changes:
```bash
uvicorn backend.main:app --reload
```

## Troubleshooting

### Issue: Qwen3 model won't download

**Solution**: Ensure you have ~1GB free disk space. Model downloads to `~/.cache/huggingface/`

### Issue: ArXiv API errors (429, timeouts)

**Solution**: The client includes rate limiting (3 sec delay). If you still get errors, increase `arxiv_rate_limit` in config.

### Issue: No papers accepted in tracking

**Solution**:
1. Check logs for rejection reasons
2. Lower `threshold_moderate` in config
3. Increase `max_velocity` if velocity rejections are common

### Issue: Out of memory during embedding

**Solution**: Reduce `max_papers_per_window` in tracking request

## Validation Example

Test with known concept evolution (Transformers 2017-2018):

```bash
curl -X POST "http://localhost:8000/api/track" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_paper_ids": ["1706.03762"],
    "end_date": "2018-06-30",
    "window_months": 6,
    "max_papers_per_window": 50
  }'
```

**Expected**:
- Should find BERT-related papers (1810.04805)
- Should find other transformer variants
- Should NOT jump to unrelated NLP (pure RNN papers)
- Similarity should stay above 0.65
- 2-3 time steps with 10-30 papers each

## Future Enhancements (Post-MVP)

- ✅ **Linear tracking** (current MVP)
- 🔲 **Tree branching** with HDBSCAN clustering
- 🔲 **Web UI** with D3.js visualization
- 🔲 **Bidirectional tracking** (trace concepts to their origins)
- 🔲 **Multi-signal validation** (citations, author overlap)

## License

MIT License - See LICENSE file

## Contributing

This is an MVP/prototype. For issues or suggestions, please open an issue on GitHub.

## Acknowledgments

- ArXiv for open access to research papers
- Qwen team for the embedding model
- FastAPI and sentence-transformers communities
