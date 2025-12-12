# ArXiv Concept Evolution Tracker - Requirements & Concept Document

## Project Vision

Build a web application that tracks how scientific concepts evolve over time by analyzing ArXiv papers through semantic embeddings. The system will allow users to seed a concept with initial papers and watch how that concept develops, shifts terminology, and spawns related ideas across years of research.

## Problem Statement

Researchers studying the evolution of ideas in machine learning face several challenges:
- **Citation limitations**: Traditional citation networks miss conceptually similar papers that don't cite each other
- **Terminology drift**: Concepts evolve their terminology ("attention mechanism" → "self-attention" → "flash attention")
- **Manual effort**: Literature reviews require manually reading hundreds of papers
- **Temporal blindness**: Hard to see when concepts emerged, peaked, or transformed

This tool addresses these by using semantic embeddings to track concept evolution independent of citations or exact terminology matches.

## Core Functionality (MVP)

### 1. Seed Paper Discovery
- **Search ArXiv**: Users search by keywords to find potential seed papers
- **Browse results**: Display paper metadata (title, authors, abstract, date, categories)
- **Select seeds**: Choose 1-5 papers that represent the starting concept

### 2. Concept Tracking
- **Forward tracking**: From seed papers forward in time to present day
- **Temporal windows**: Track in configurable windows (default: 6 months)
- **Multi-signal validation**: Use embedding similarity + velocity constraints + metadata
- **Kalman filtering**: Physics-inspired constraints prevent sudden concept jumps
- **Branch detection**: Identify when concept splits into multiple directions
- **Tree structure**: Build recursive tree showing concept evolution and branching

### 3. Evolution Tree Visualization
- **Tree structure**: Show branching evolution, not just linear timeline
- **Branch points**: Identify when/where concepts diverged
- **Path exploration**: Click any branch to explore that direction
- **Node details**: See papers at each node, confidence scores, descriptions
- **Leaf branches**: Highlight current active endpoints

## User Stories

### Primary User: ML Researcher conducting literature review

**Story 1: Understanding transformer evolution**
> "As a researcher new to transformers, I want to see how the concept evolved from 2017 to 2024, so I can understand the key innovations and current state of the art."

- User searches "attention is all you need"
- Selects Vaswani et al. 2017 paper as seed
- System tracks forward showing evolution tree with branches:
  - Main trunk: Original transformer architecture
  - Branch 1 (2018): BERT-style bidirectional models
  - Branch 2 (2019): GPT-style autoregressive models
  - Branch 3 (2020): Vision transformers
  - Sub-branches: Flash attention, sparse attention variants
- User explores each branch to understand different directions
- System highlights that concept split into distinct research directions

**Story 2: Exploring diffusion models**
> "As someone interested in diffusion models, I want to trace their development from early work to Stable Diffusion, so I can identify key papers I should read."

- User searches "diffusion models" or "denoising diffusion"
- Selects 2-3 early papers as seeds
- System shows progression through DDPM, DDIM, Stable Diffusion, consistency models
- User identifies 20 most important papers to read

**Story 3: Discovering understudied areas**
> "As a PhD student looking for research directions, I want to find concepts that were promising but didn't develop much, so I can identify potential research opportunities."

- User tracks a concept that peaks then declines
- System shows where papers dropped off
- User investigates why the area stalled

## Technical Requirements

### Functional Requirements

**FR1: ArXiv API Integration**
- Search ArXiv papers by keyword/query
- Retrieve paper metadata: title, abstract, authors, date, categories, ArXiv ID
- Handle API rate limits gracefully
- Support date range filtering

**FR2: Embedding Generation**
- Generate embeddings from paper title + abstract
- Use models with 32K+ context window (Voyage-3-lite or Qwen3-Embedding)
- Cache embeddings to avoid recomputation
- Handle embedding API failures gracefully

**FR3: Concept Tracking Algorithm**
- Initialize tracker with seed papers
- Step through time windows (configurable duration)
- Evaluate candidate papers using multi-signal validation:
  - Embedding similarity (primary signal, high thresholds: 0.85+ for auto-include)
  - Velocity constraints (Kalman filter: prevent impossible concept jumps)
  - Author overlap (from ArXiv metadata)
  - Category consistency (ArXiv categories)
- Apply physics-inspired constraints:
  - Maximum velocity: concepts can't drift faster than threshold
  - Maximum acceleration: rate of change must be plausible
- Detect branches using clustering when papers split into distinct groups
- Build tree structure with parent-child relationships
- Track multiple branches simultaneously
- Record tree nodes with papers, vectors, confidence scores, and branch descriptions

**FR4: API Endpoints**
- `POST /api/search`: Search ArXiv papers
- `POST /api/track`: Track concept evolution from seeds
- `GET /api/paper/{arxiv_id}`: Get paper details
- All endpoints return JSON

**FR5: Web Interface**
- Seed paper search and selection interface
- Tree visualization showing concept evolution with branches
- Interactive node exploration (click to see branch details)
- Branch point indicators showing where concepts diverged
- Node detail panels with papers, confidence scores, descriptions
- Paper cards with title, authors, abstract, similarity score, confidence
- Direct links to ArXiv PDFs
- Path highlighting from seed to current branches

### Non-Functional Requirements

**NFR1: Performance**
- Search results return within 2 seconds
- Tracking completes within 60 seconds for 5-year span (includes branching detection)
- Frontend renders tree within 2 seconds
- Branch detection runs efficiently on 500+ papers per window

**NFR2: Usability**
- Clear visual feedback during long-running operations
- Intuitive seed selection (checkbox or multi-select)
- Readable timeline with appropriate density
- Mobile-friendly responsive design (stretch goal)

**NFR3: Reliability**
- Handle ArXiv API timeouts/errors
- Validate user inputs
- Provide meaningful error messages
- Log errors for debugging

**NFR4: Maintainability**
- Clean separation of concerns (API, tracking logic, frontend)
- Comprehensive docstrings
- Type hints throughout Python code
- Modular components for future extension

## Success Criteria

### MVP Success
- [ ] User can search ArXiv and find papers on any topic
- [ ] User can select 1-5 seed papers
- [ ] System tracks concept forward through time with branching
- [ ] Tree visualization shows clear evolution with branch points
- [ ] User can explore individual branches and nodes
- [ ] System correctly identifies major branches (e.g., BERT vs GPT split)
- [ ] Kalman filter prevents unrealistic concept jumps
- [ ] Total development time: 2-3 weeks

### Validation
- Track "transformer" from 2017-2024 and verify it:
  - Finds BERT, GPT series, Vision Transformers
  - Correctly identifies branch point around 2018-2019
  - Separates encoder-only, decoder-only, encoder-decoder variants
- Track "diffusion models" and verify branching into image/video/audio variants
- Kalman filter rejects papers with impossible acceleration
- User testing with 3-5 ML researchers
- Positive feedback on branch detection accuracy

## Explicitly Out of Scope (v1)

The following are valuable but deferred to future iterations:

### Not in MVP
- ❌ Bidirectional tracking (backward to origins)
- ❌ Multi-scale tracking (simultaneous tracking at different time scales)
- ❌ PDF download and section-level indexing
- ❌ Citation network integration (unless using Semantic Scholar API)
- ❌ 2D embedding space visualization (UMAP/t-SNE plots)
- ❌ Advanced terminology extraction and evolution tracking
- ❌ User accounts and saved searches
- ❌ Batch processing multiple concepts
- ❌ Comparison of multiple concept trajectories
- ❌ Paper recommendation system
- ❌ Interactive branch pruning/merging

### Future Enhancements (v2+)
- Bidirectional tracking for finding origins
- Citation integration via Semantic Scholar API
- Section-level embeddings (intro/methods/results)
- Advanced terminology evolution visualization
- Predictive tracking (where concept might go next)
- User feedback loop (mark papers as relevant/irrelevant to improve tracking)
- Collaborative features (share tracked trees)
- Tree comparison (compare evolution of different concepts)
- Branch similarity analysis (show when different branches converge)
- Export tree as JSON/interactive visualization

## Technical Constraints

### Platform
- Backend: Python 3.10+
- Frontend: Modern browsers (Chrome, Firefox, Safari, Edge)
- Development: Mac Studio M4 Max with 128GB RAM

### Data Sources
- **ArXiv**: Open access, unlimited metadata, rate-limited API
- **Embeddings**: Voyage-3-lite API (recommended) or Qwen3-Embedding (local)

### Limitations
- **Metadata only**: MVP uses title + abstract, not full text
- **ArXiv only**: Does not include papers from other sources
- **English only**: No multilingual support in v1
- **Recent papers**: ArXiv metadata complete from ~2007 onward

## Key Assumptions

1. **Title + abstract sufficient**: Embeddings from title/abstract capture enough semantic information for concept tracking
2. **High thresholds work**: Similarity threshold of 0.85+ accurately identifies related papers with high confidence
3. **Velocity constraints realistic**: Kalman filter constraints (max velocity, max acceleration) prevent false positives without missing true evolution
4. **Clustering detects branches**: HDBSCAN or similar can reliably identify when papers split into distinct conceptual groups
5. **6-month windows appropriate**: Default window size captures meaningful evolution steps and allows for branch detection
6. **Metadata signals helpful**: Author overlap and category consistency provide useful validation signals
7. **API costs manageable**: Voyage-3-lite at $0.02/1M tokens = ~$2 for 100K papers
8. **Tree structure interpretable**: Users can understand and navigate tree-based concept evolution

## Dependencies

### Python Packages
- `arxiv` - ArXiv API wrapper
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `numpy` - Numerical operations
- `scikit-learn` - Cosine similarity, HDBSCAN clustering
- `voyageai` or `sentence-transformers` - Embeddings
- `python-dotenv` - Environment configuration
- `requests` - HTTP client (for Semantic Scholar API if used)

### Frontend
- Vanilla JavaScript or React (TBD)
- Plotly.js - Interactive charts
- Tailwind CSS - Styling (optional)

## Risk Analysis

### Technical Risks

**Risk 1: Kalman filter tuning**
- *Description*: Velocity/acceleration constraints too strict (miss true evolution) or too loose (allow false positives)
- *Likelihood*: Medium
- *Impact*: High
- *Mitigation*: Test with known evolutions (transformers), provide adjustable parameters, validate with domain experts

**Risk 2: Branch detection accuracy**
- *Description*: Clustering either over-splits (too many false branches) or under-splits (misses real divergence)
- *Likelihood*: Medium
- *Impact*: Medium
- *Mitigation*: Test with known branching concepts, tune clustering parameters, allow user to merge/split branches

**Risk 3: Embedding quality**
- *Description*: Title+abstract embeddings don't capture enough semantic detail
- *Likelihood*: Low
- *Impact*: High
- *Mitigation*: Use state-of-the-art 32K context models, test with known concept evolutions

**Risk 4: Computational complexity**
- *Description*: Tree tracking with multiple branches and Kalman filtering is too slow
- *Likelihood*: Low
- *Impact*: Medium
- *Mitigation*: Efficient numpy operations, limit max branches, cache aggressively

**Risk 5: API rate limits**
- *Description*: ArXiv API throttles requests during development/testing
- *Likelihood*: Medium
- *Impact*: Low
- *Mitigation*: Implement exponential backoff, respect rate limits, cache aggressively

### Product Risks

**Risk 5: User confusion**
- *Description*: Seed selection or interpretation unclear to users
- *Likelihood*: Medium
- *Impact*: Medium
- *Mitigation*: Clear UI/UX, good examples, tooltips, documentation

**Risk 6: Validation difficulty**
- *Description*: Hard to verify tracking results are "correct"
- *Likelihood*: Medium
- *Impact*: Medium
- *Mitigation*: Test with well-known concepts (transformers, diffusion), manual validation by domain experts

## Measurement & Metrics

### Development Metrics
- Lines of code
- Test coverage (target: >80% for core tracking logic)
- API endpoint response times
- Embedding generation time per paper

### Usage Metrics (post-deployment)
- Number of concept tracks created
- Average tracking duration
- Most searched seed papers
- Most tracked concepts
- User session duration

### Quality Metrics
- Tracking accuracy for known concepts (manual validation)
- User satisfaction scores (survey)
- Number of papers found vs. expected (for known evolutions)

## Glossary

- **Seed paper**: Initial paper(s) used to start concept tracking
- **Concept vector**: Embedding vector representing the current state of the concept
- **Velocity**: Rate of change of concept vector through embedding space (Kalman filter)
- **Acceleration**: Rate of change of velocity (physics constraint to prevent jumps)
- **Time window**: Fixed duration period for collecting related papers (e.g., 6 months)
- **Similarity threshold**: Minimum cosine similarity required to consider a paper related
  - High threshold (0.85+): Auto-include with high confidence
  - Medium threshold (0.75-0.85): Include if passes velocity check
  - Low threshold (0.65-0.75): Include if multiple signals confirm
- **Kalman filter**: Physics-inspired algorithm for smooth state estimation with constraints
- **Branch point**: Node in tree where concept splits into multiple directions
- **Branch**: Separate evolutionary path that diverges from parent concept
- **Tree node**: Single state in concept evolution with papers, vector, and metadata
- **Leaf node**: Current endpoint of a branch (no children)
- **Confidence score**: Combined metric indicating how certain we are about paper relevance
- **Innovation**: Difference between observed paper and predicted concept state
- **Clustering**: Algorithm to detect when papers split into distinct groups (HDBSCAN)

## References

- ArXiv API Documentation: https://info.arxiv.org/help/api/
- Voyage AI Embeddings: https://www.voyageai.com/
- Qwen3-Embedding: https://huggingface.co/Qwen/Qwen3-Embedding
- Cosine Similarity: Standard metric for embedding similarity
