# ArXiv Concept Tracker - AI Assistant Context

## Project Overview

**Type**: Portfolio/showcase project for training course
**Timeline**: 1-2 days development remaining
**Goal**: Demonstrate full-stack ML/NLP capabilities with deployable demo

This system tracks the evolution of research concepts over time by analyzing arXiv papers. Starting from seed papers, it uses a Kalman filter-inspired approach to follow how concepts drift and develop through the research literature.

**Core Innovation**: Uses embedding space velocity and acceleration constraints to ensure smooth, realistic concept evolution rather than random topic drift.

**Current Status**: Backend complete and validated. Need: web frontend + Render.com deployment.

## Architecture Summary

### Components

```
┌─────────────────┐
│   FastAPI App   │  Entry point, REST API
│   (main.py)     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ ConceptTracker  │  Orchestrates tracking across time windows
│  (tracker.py)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    v         v
┌─────────┐ ┌──────────────────┐
│ ArXiv   │ │ KalmanTracker    │  Physics-inspired constraints
│ Client  │ │ (kalman_tracker) │  - Velocity limits
└─────────┘ └──────────────────┘  - Acceleration limits
    │               │
    v               v
┌──────────────────────┐
│ EmbeddingService     │  Qwen/Qwen3-Embedding-0.6B
│ (embedding_service)  │  Caching layer
└──────────────────────┘
```

## Key Files

### Core Logic
- **`backend/main.py`**: FastAPI application, API endpoints
- **`backend/tracker.py`**: `ConceptTracker` - orchestrates time-windowed tracking
- **`backend/kalman_tracker.py`**: `KalmanConceptTracker` - physics-inspired filtering with velocity/acceleration constraints
- **`backend/arxiv_client.py`**: ArXiv API wrapper with rate limiting
- **`backend/embedding_service.py`**: Embedding generation and caching
- **`backend/models.py`**: Pydantic models for request/response

### Configuration
- **`backend/config.py`**: Default settings (can be overridden by `.env`)
- **`.env`**: Runtime configuration (gitignored, see `.env.example`)

### Development
- **`.idea/runConfigurations/`**: PyCharm debug configurations
- **`debug_track_request.py`**: Test client for debugging

## How Concept Tracking Works

### The Kalman Filter Analogy

The tracker models concept evolution in embedding space using physics-inspired constraints:

1. **Position**: Current concept vector (normalized embedding)
2. **Velocity**: Rate and direction of concept drift
3. **Acceleration**: Change in velocity (prevents erratic jumps)

### Mathematical Foundation

For **normalized embedding vectors**, the relationship between similarity and velocity is:

```
velocity = ||v_new - v_current|| = √(2 - 2×similarity)
```

This means:
- similarity = 1.0 → velocity = 0.0 (identical)
- similarity = 0.5 → velocity = 1.0
- similarity = 0.0 → velocity = 1.414 (orthogonal)

**Key Insight**: `max_velocity` directly determines the minimum similarity accepted!

### First Step Special Case

On the first step after initialization, velocity is zero, so:
```
acceleration = velocity_change = implied_velocity - 0 = implied_velocity
```

Therefore, **acceleration check is skipped on the first step** (see `kalman_tracker.py:99-107`).

## Configuration & Tuning

### Critical Parameters (`.env`)

```bash
# Velocity constraint (most important!)
MAX_VELOCITY=1.0        # Allows similarity >= 0.50

# Acceleration constraint (controls smoothness)
MAX_ACCELERATION=0.9    # Lower = smoother evolution

# Similarity thresholds
THRESHOLD_REJECT=0.40   # Hard minimum
THRESHOLD_MODERATE=0.50
THRESHOLD_STRONG=0.60
THRESHOLD_AUTO_INCLUDE=0.85
```

### Tuning Guidelines

**If getting too few papers:**
- Increase `MAX_VELOCITY` (most impactful)
- Increase `MAX_ACCELERATION`
- Lower `THRESHOLD_REJECT`

**If concept drifts too erratically:**
- Decrease `MAX_ACCELERATION`
- Increase similarity thresholds

**If concept evolves too slowly:**
- Increase `MAX_VELOCITY`
- Increase `MAX_ACCELERATION`

### Confidence Tiers

Papers are categorized by similarity (NOT by confidence scores):
- **High**: similarity >= 0.70
- **Moderate**: similarity 0.60-0.70
- **Low**: similarity < 0.60

Note: All accepted papers are relevant; "Low" doesn't mean bad!

## Development Setup

### PyCharm Debugging

**Important**: Use the **"FastAPI with Uvicorn"** run configuration for debugging, NOT "FastAPI Server".

- `FastAPI with Uvicorn`: Runs `python -m uvicorn backend.main:app` (debugger compatible)
- `FastAPI Server`: Runs `python backend/main.py` (has asyncio conflicts with debugger)

**To debug server code:**
1. Select "FastAPI with Uvicorn" configuration
2. Click Debug (🐛) not Run (▶️)
3. Set breakpoints in `backend/` files
4. Run `debug_track_request.py` to trigger requests

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_kalman.py -v

# With coverage
pytest --cov=backend --cov-report=html
```

## Common Tasks

### Adding a New Endpoint

1. Add Pydantic models to `backend/models.py`
2. Add endpoint to `backend/main.py`
3. Add business logic to appropriate service file
4. Update tests

### Changing Embedding Model

Edit `.env`:
```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Cache will be rebuilt automatically (stored in `./cache/`).

### Debugging Concept Drift

1. Enable DEBUG logging: `LOG_LEVEL=DEBUG` in `.env`
2. Look for these key log messages:
   - `"Position drift: X.XXXX"` - how much concept moved
   - `"Velocity magnitude: X.XXX"` - rate of change
   - `"Acceleration magnitude: X.XXX"` - smoothness
   - `"Rejected - below X.XXX"` or `"Invalid - confidence: 0.000"` - why papers rejected

3. Check similarity distribution in results
4. Adjust `MAX_VELOCITY` and `MAX_ACCELERATION` accordingly

## Validation Strategy

### How to Verify Tracking Quality

1. **Check average similarity progression**: Should increase or stabilize over time
2. **Inspect paper titles/abstracts**: Should follow a coherent thematic thread
3. **Verify with external model**: Ask GPT-4/Gemini if the progression makes sense
4. **Look for seminal papers**: Major papers in the concept should appear or be cited

### Expected Behavior

- **Similarity starts low** (0.45-0.50) as tracker explores
- **Similarity increases** (0.60-0.70) as concept converges
- **Position drift decreases** over time as concept stabilizes
- **Papers show thematic progression** (e.g., Transformers → BERT → Efficient Transformers)

## Known Issues & Gotchas

### 1. Original Default Values Were Unrealistic

The original defaults (`max_velocity=0.05`, `max_acceleration=0.02`) were far too restrictive, only accepting similarity > 0.998. These have been updated to realistic values.

### 2. Confidence vs Similarity Confusion

- **Similarity**: Raw cosine similarity between embeddings (0-1)
- **Confidence**: Similarity with acceleration penalties applied (used internally)
- **Tiers**: Based on similarity, not confidence (more intuitive)

### 3. Missing Seminal Papers

A tracker might miss important papers (e.g., BERT) if they fall:
- At time window boundaries
- Slightly below threshold at that specific step

However, if papers **citing** the seminal work appear, the tracker is working correctly.

### 4. arXiv Rate Limiting

The arXiv API has a 3-second rate limit. Large queries take time. Be patient!

### 5. Cache Invalidation

Embedding cache is keyed by model name + text. Changing the model doesn't auto-clear old cache. Manually delete `./cache/` if switching models.

## Recent Changes

### 2024-12-15: Major Improvements

1. **First-step acceleration skip**: Acceleration check now skipped when velocity is zero (first step after init)
2. **Realistic defaults**: Updated `config.py` with `max_velocity=1.0`, `max_acceleration=0.6`
3. **Position drift logging**: Now reported at INFO level alongside similarity
4. **Fixed confidence tiers**: Now based on similarity (0.70, 0.60) instead of hardcoded confidence values
5. **Better documentation**: Added math explanation and tuning guidelines

## What's Next (See PLAN.md)

**Immediate priorities for showcase**:
1. Web frontend with timeline visualization
2. Render.com deployment
3. README with screenshots and live demo link

**Not building** (out of scope for showcase):
- User accounts or persistence
- Branching detection (too complex)
- Production features

## Debugging Checklist

When concept tracking isn't working as expected:

- [ ] Check `MAX_VELOCITY` allows your target similarity range
- [ ] Verify `THRESHOLD_REJECT` isn't too high
- [ ] Look at position drift - is it reasonable?
- [ ] Check velocity magnitude - is it being clipped?
- [ ] Review rejected papers - why were they rejected?
- [ ] Validate with a small, known dataset first
- [ ] Compare similarity distribution across steps
- [ ] Ask an external LLM to review the paper progression

## Contact & Resources

- **arXiv API**: https://arxiv.org/help/api/
- **Qwen Embeddings**: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- **FastAPI Docs**: https://fastapi.tiangolo.com/
