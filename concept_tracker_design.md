# ArXiv Concept Evolution Tracker - Design Document

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Browser)                    │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  Seed Selection │  │    Tree Visualization            │ │
│  │    Interface    │  │  (D3.js/Plotly tree + branches)  │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/JSON
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Python)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   Search     │  │   Tracking   │  │   Tree Export   │ │
│  │  Endpoint    │  │   Endpoint   │  │    Endpoint     │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└──────┬──────────────────┬────────────────────┬─────────────┘
       │                  │                    │
       ▼                  ▼                    ▼
┌──────────────┐   ┌─────────────────┐  ┌──────────────────┐
│  ArXiv API   │   │  ConceptTree +  │  │ Embedding Cache │
│   Wrapper    │   │ KalmanTracker   │  │  (pickle/json)  │
└──────┬───────┘   └────────┬────────┘  └─────────┬────────┘
       │                    │                      │
       ▼                    ▼                      ▼
┌──────────────┐   ┌─────────────────┐  ┌──────────────────┐
│   arxiv.py   │   │  HDBSCAN        │  │  Voyage API or  │
│   library    │   │  clustering     │  │  Qwen3 local    │
└──────────────┘   └─────────────────┘  └──────────────────┘
```

### Component Overview

**Frontend Layer**
- Static HTML/CSS/JS served by FastAPI
- Responsible for user interaction and visualization
- Makes API calls to backend

**API Layer** 
- FastAPI REST endpoints
- Request validation with Pydantic
- Response formatting to JSON

**Business Logic Layer**
- ConceptTracker class (core algorithm)
- ArXiv data fetching and parsing
- Embedding generation and caching

**External Services**
- ArXiv API for paper metadata
- Voyage AI or local Qwen3 for embeddings

## Data Models

### Paper Model

```python
from pydantic import BaseModel
from datetime import datetime

class Paper(BaseModel):
    """Represents an ArXiv paper"""
    arxiv_id: str              # e.g., "1706.03762"
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    updated: datetime | None
    categories: list[str]      # e.g., ["cs.LG", "cs.CL"]
    pdf_url: str              # Direct link to PDF
    
    # Computed fields (not in API response)
    embedding: list[float] | None = None
    similarity: float | None = None
```

### TimelineStep Model

```python
class ConceptNode(BaseModel):
    """Represents one node in the concept evolution tree"""
    node_id: str
    parent_id: str | None
    
    # Temporal
    start_date: datetime
    end_date: datetime
    
    # Papers and concept state
    papers: list[Paper]
    num_papers: int
    concept_vector: list[float]
    velocity: list[float]       # Rate of change (Kalman filter)
    
    # Branch metadata
    description: str            # e.g., "BERT-style bidirectional models"
    is_branch_point: bool      # Does this node have multiple children?
    branch_id: str             # Unique ID for this branch path
    
    # Confidence and validation
    avg_similarity: float
    confidence_score: float
    num_high_confidence: int   # Papers with sim > 0.85
    num_ambiguous: int         # Papers with sim 0.65-0.85
    
    # Kalman filter state
    drift_magnitude: float     # Distance from parent
    acceleration: float        # Change in velocity
```

### TrackingRequest Model

```python
class TrackingRequest(BaseModel):
    """Request to track concept evolution"""
    seed_paper_ids: list[str]  # ArXiv IDs
    end_date: str              # ISO format: "2025-01-01"
    window_months: int = 6     # Default 6-month windows
    similarity_threshold: float = 0.65
    max_papers_per_window: int = 50
```

### TrackingResponse Model

```python
class BranchInfo(BaseModel):
    """Information about an active branch endpoint"""
    branch_id: str
    description: str
    num_papers: int
    last_update: datetime
    sample_papers: list[Paper]  # Top 5 representative papers

class TreeNodeResponse(BaseModel):
    """Recursive tree node for API response"""
    node_id: str
    parent_id: str | None
    description: str
    date_range: tuple[str, str]
    num_papers: int
    is_branch_point: bool
    confidence_score: float
    children: list['TreeNodeResponse']  # Recursive structure
    sample_papers: list[Paper]  # Top papers at this node

class TrackingResponse(BaseModel):
    """Response containing concept evolution tree"""
    tree_root: TreeNodeResponse
    seed_papers: list[Paper]
    
    # Summary statistics
    total_nodes: int
    total_papers_tracked: int
    num_branches: int
    max_depth: int
    
    # Active branches (leaf nodes)
    active_branches: list[BranchInfo]
    
    # Date range
    date_range: tuple[str, str]
    
    # Tracking metadata
    window_months: int
    similarity_threshold: float
```

## Core Components

### 1. ArXiv API Wrapper

**Purpose**: Fetch paper metadata from ArXiv

**Key Methods**:
```python
class ArXivClient:
    def __init__(self, rate_limit_delay: float = 3.0):
        """Initialize with rate limiting"""
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=rate_limit_delay,
            num_retries=3
        )
    
    def search_papers(
        self, 
        query: str, 
        start_date: str | None = None,
        end_date: str | None = None,
        max_results: int = 100
    ) -> list[Paper]:
        """Search ArXiv with optional date filtering"""
        pass
    
    def get_paper_by_id(self, arxiv_id: str) -> Paper:
        """Fetch single paper by ArXiv ID"""
        pass
    
    def get_papers_by_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        """Batch fetch multiple papers"""
        pass
```

**Error Handling**:
- Retry on network failures (3 attempts)
- Log rate limit hits
- Return partial results on timeout
- Clear error messages for invalid IDs

### 2. Embedding Service

**Purpose**: Generate and cache embeddings

**Key Methods**:
```python
class EmbeddingService:
    def __init__(self, model: str = "voyage-3-lite", cache_path: str = "./cache"):
        """Initialize embedding model and cache"""
        self.model = model
        self.cache = EmbeddingCache(cache_path)
    
    def embed_paper(self, paper: Paper) -> np.ndarray:
        """Generate embedding from title + abstract"""
        # Check cache first
        if cached := self.cache.get(paper.arxiv_id):
            return cached
        
        # Generate embedding
        text = f"{paper.title} {paper.abstract}"
        embedding = self._generate_embedding(text)
        
        # Cache result
        self.cache.set(paper.arxiv_id, embedding)
        return embedding
    
    def embed_papers(self, papers: list[Paper]) -> np.ndarray:
        """Batch embed multiple papers"""
        pass
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Call embedding API/model"""
        pass
```

**Caching Strategy**:
- Cache embeddings as pickle files: `cache/{arxiv_id}.pkl`
- Cache metadata as JSON: `cache/metadata.json`
- LRU eviction if cache exceeds size limit (optional for MVP)

**Model Selection**:
```python
# Option A: Voyage API (recommended for MVP)
import voyageai
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
embedding = vo.embed([text], model="voyage-3-lite").embeddings[0]

# Option B: Local Qwen3 (future)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Qwen/Qwen3-Embedding')
embedding = model.encode(text)
```

### 3. Kalman Concept Tracker

**Purpose**: Track concept evolution with physics-inspired velocity constraints

**Key Methods**:
```python
class KalmanConceptTracker:
    """
    Track concept evolution with velocity and acceleration constraints
    Prevents unrealistic jumps through embedding space
    """
    def __init__(
        self,
        embedding_dim: int = 1024,
        max_velocity: float = 0.05,
        max_acceleration: float = 0.02
    ):
        # State
        self.position: np.ndarray | None = None  # Current concept vector
        self.velocity: np.ndarray | None = None  # Rate of change
        
        # Kalman filter parameters
        self.process_noise = 0.01
        self.measurement_noise = 0.1
        
        # Physics constraints
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        
        # Confidence thresholds
        self.thresholds = {
            'auto_include': 0.85,   # Very high confidence
            'strong': 0.75,         # Likely related
            'moderate': 0.65,       # Needs validation
            'reject': 0.55          # Too dissimilar
        }
    
    def initialize(self, seed_papers: list[Paper]) -> None:
        """Initialize with seed papers"""
        embeddings = np.array([p.embedding for p in seed_papers])
        self.position = embeddings.mean(axis=0)
        self.position /= np.linalg.norm(self.position)
        self.velocity = np.zeros_like(self.position)
    
    def predict(self, time_delta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Predict next state using current velocity"""
        predicted_pos = self.position + self.velocity * time_delta
        predicted_pos /= np.linalg.norm(predicted_pos)
        return predicted_pos, self.velocity
    
    def evaluate_candidate(
        self, 
        candidate_vector: np.ndarray
    ) -> tuple[bool, float, str]:
        """
        Check if candidate satisfies velocity constraints
        Returns: (is_valid, confidence, reason)
        """
        # Embedding similarity
        similarity = cosine_similarity([self.position], [candidate_vector])[0][0]
        
        if similarity < self.thresholds['reject']:
            return False, 0.0, "Below similarity threshold"
        
        # Check velocity constraints
        implied_velocity = candidate_vector - self.position
        velocity_magnitude = np.linalg.norm(implied_velocity)
        
        if velocity_magnitude > self.max_velocity:
            return False, 0.0, f"Velocity {velocity_magnitude:.3f} exceeds max {self.max_velocity}"
        
        # Check acceleration constraints
        velocity_change = implied_velocity - self.velocity
        acceleration = np.linalg.norm(velocity_change)
        
        if acceleration > self.max_acceleration:
            return False, 0.0, f"Acceleration {acceleration:.3f} exceeds max {self.max_acceleration}"
        
        # Calculate confidence based on similarity and constraints
        if similarity >= self.thresholds['auto_include']:
            confidence = 0.95
        elif similarity >= self.thresholds['strong']:
            confidence = 0.80
        elif similarity >= self.thresholds['moderate']:
            confidence = 0.65
        else:
            confidence = 0.50
        
        # Reduce confidence for high acceleration
        confidence *= (1.0 - acceleration / self.max_acceleration)
        
        return True, confidence, "Passes all constraints"
    
    def update(self, measurement_vector: np.ndarray, confidence: float) -> None:
        """Kalman update with new measurement"""
        # Kalman gain
        kalman_gain = confidence * 0.3  # Blend factor
        
        # Update position
        innovation = measurement_vector - self.position
        self.position = self.position + kalman_gain * innovation
        self.position /= np.linalg.norm(self.position)
        
        # Update velocity (smoothed)
        new_velocity = innovation
        self.velocity = 0.8 * self.velocity + 0.2 * new_velocity
        
        # Clip velocity
        velocity_mag = np.linalg.norm(self.velocity)
        if velocity_mag > self.max_velocity:
            self.velocity *= (self.max_velocity / velocity_mag)
    
    def process_papers(
        self, 
        papers: list[Paper]
    ) -> tuple[list[tuple], list[tuple], list[tuple]]:
        """
        Evaluate all papers in window
        Returns: (accepted, rejected, ambiguous)
        """
        accepted = []
        rejected = []
        ambiguous = []
        
        for paper in papers:
            similarity = cosine_similarity([self.position], [paper.embedding])[0][0]
            
            if similarity < self.thresholds['reject']:
                rejected.append((paper, similarity, 0.0, "Too dissimilar"))
                continue
            
            is_valid, confidence, reason = self.evaluate_candidate(paper.embedding)
            
            if not is_valid:
                rejected.append((paper, similarity, confidence, reason))
            elif similarity >= self.thresholds['auto_include']:
                accepted.append((paper, similarity, confidence, "High confidence"))
            elif similarity >= self.thresholds['strong'] and confidence > 0.5:
                accepted.append((paper, similarity, confidence, "Moderate confidence"))
            elif similarity >= self.thresholds['moderate']:
                ambiguous.append((paper, similarity, confidence, "Needs review"))
            else:
                rejected.append((paper, similarity, confidence, "Low confidence"))
        
        return accepted, rejected, ambiguous


### 4. Concept Tree

**Purpose**: Manage tree structure with branching

**Implementation**:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConceptNode:
    """Node in concept evolution tree"""
    node_id: str
    parent: Optional['ConceptNode']
    children: list['ConceptNode']
    
    # Temporal and spatial
    date_range: tuple[datetime, datetime]
    papers: list[Paper]
    concept_vector: np.ndarray
    velocity: np.ndarray
    
    # Metadata
    description: str
    num_papers: int
    confidence: float
    is_branch_point: bool

class ConceptTree:
    """Manages tree structure and branching"""
    def __init__(self, root_papers: list[Paper]):
        embeddings = np.array([p.embedding for p in root_papers])
        
        self.root = ConceptNode(
            node_id="root",
            parent=None,
            children=[],
            date_range=(
                min(p.published for p in root_papers),
                max(p.published for p in root_papers)
            ),
            papers=root_papers,
            concept_vector=embeddings.mean(axis=0),
            velocity=np.zeros(len(embeddings[0])),
            description="Seed concept",
            num_papers=len(root_papers),
            confidence=1.0,
            is_branch_point=False
        )
        
        self.all_nodes = [self.root]
        self.node_counter = 0
    
    def detect_branches(
        self, 
        papers: list[Paper], 
        min_cluster_size: int = 5
    ) -> Optional[list[list[Paper]]]:
        """Use HDBSCAN to detect if papers split into clusters"""
        if len(papers) < min_cluster_size * 2:
            return None
        
        from sklearn.cluster import HDBSCAN
        
        embeddings = np.array([p.embedding for p in papers])
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=3)
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters <= 1:
            return None
        
        # Group papers by cluster
        clusters = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_papers = [p for p, l in zip(papers, labels) if l == cluster_id]
            if len(cluster_papers) >= min_cluster_size:
                clusters.append(cluster_papers)
        
        return clusters if len(clusters) > 1 else None
    
    def grow_branch(
        self,
        parent: ConceptNode,
        papers: list[Paper],
        window_end: datetime
    ) -> list[ConceptNode]:
        """Grow tree, potentially creating branches"""
        branches = self.detect_branches(papers)
        
        if branches is None:
            # Single continuation
            child = self._create_node(parent, papers, window_end, "Continuation")
            parent.children.append(child)
            self.all_nodes.append(child)
            return [child]
        
        # Multiple branches detected
        parent.is_branch_point = True
        children = []
        
        for i, branch_papers in enumerate(branches):
            desc = self._describe_branch(branch_papers)
            child = self._create_node(parent, branch_papers, window_end, desc)
            parent.children.append(child)
            self.all_nodes.append(child)
            children.append(child)
        
        return children
    
    def _create_node(
        self, 
        parent: ConceptNode, 
        papers: list[Paper],
        end_date: datetime,
        description: str
    ) -> ConceptNode:
        """Create new tree node"""
        embeddings = np.array([p.embedding for p in papers])
        new_vector = embeddings.mean(axis=0)
        new_vector /= np.linalg.norm(new_vector)
        
        self.node_counter += 1
        
        return ConceptNode(
            node_id=f"node_{self.node_counter}",
            parent=parent,
            children=[],
            date_range=(parent.date_range[1], end_date),
            papers=papers,
            concept_vector=new_vector,
            velocity=new_vector - parent.concept_vector,
            description=description,
            num_papers=len(papers),
            confidence=cosine_similarity([new_vector], [parent.concept_vector])[0][0],
            is_branch_point=False
        )
    
    def _describe_branch(self, papers: list[Paper]) -> str:
        """Generate branch description from key terms"""
        from collections import Counter
        words = []
        for p in papers[:10]:  # Sample papers
            words.extend(p.title.lower().split())
        common = Counter(words).most_common(3)
        return f"Branch: {', '.join([w for w, c in common])}"
    
    def to_dict(self) -> dict:
        """Export tree for JSON serialization"""
        def node_to_dict(node):
            return {
                'node_id': node.node_id,
                'description': node.description,
                'date_range': [d.isoformat() for d in node.date_range],
                'num_papers': node.num_papers,
                'is_branch_point': node.is_branch_point,
                'children': [node_to_dict(c) for c in node.children]
            }
        return node_to_dict(self.root)
```

### 4. API Endpoints

**Search Endpoint**:
```python
@app.post("/api/search")
async def search_papers(query: str, limit: int = 20) -> dict:
    """
    Search ArXiv papers by keyword
    
    Args:
        query: Search string (supports ArXiv query syntax)
        limit: Maximum results to return
    
    Returns:
        {"papers": [Paper, ...], "total": int}
    """
    try:
        papers = arxiv_client.search_papers(query, max_results=limit)
        return {
            "papers": [p.dict() for p in papers],
            "total": len(papers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Track Endpoint**:
```python
@app.post("/api/track")
async def track_concept(request: TrackingRequest) -> TrackingResponse:
    """
    Track concept evolution with branching
    
    Args:
        request: TrackingRequest with seed IDs and parameters
    
    Returns:
        TrackingResponse with tree structure
    """
    try:
        # Fetch seed papers
        seed_papers = arxiv_client.get_papers_by_ids(request.seed_paper_ids)
        
        # Initialize tree
        tree = ConceptTree(seed_papers)
        
        # Track with Kalman filter and branching
        trackers = {tree.root.node_id: KalmanConceptTracker()}
        trackers[tree.root.node_id].initialize(seed_papers)
        
        active_nodes = [tree.root]
        current_date = tree.root.date_range[1]
        end_date = datetime.fromisoformat(request.end_date)
        
        while current_date < end_date:
            window_end = current_date + timedelta(days=30 * request.window_months)
            if window_end > end_date:
                window_end = end_date
            
            # Get candidates
            candidates = arxiv_client.search_papers(
                query="cat:cs.LG OR cat:cs.CL",
                start_date=current_date.isoformat(),
                end_date=window_end.isoformat(),
                max_results=500
            )
            
            # Track each active branch
            new_active_nodes = []
            for node in active_nodes:
                tracker = trackers[node.node_id]
                
                # Evaluate with Kalman filter
                accepted, rejected, ambiguous = tracker.process_papers(candidates)
                
                if not accepted:
                    continue  # Branch died
                
                accepted_papers = [p for p, s, c, r in accepted]
                
                # Grow tree (may branch)
                children = tree.grow_branch(node, accepted_papers, window_end)
                
                # Create trackers for children
                for child in children:
                    new_tracker = KalmanConceptTracker()
                    new_tracker.position = child.concept_vector
                    new_tracker.velocity = child.velocity
                    trackers[child.node_id] = new_tracker
                    new_active_nodes.append(child)
            
            active_nodes = new_active_nodes
            current_date = window_end
        
        # Format response
        return TrackingResponse(
            tree_root=_node_to_response(tree.root),
            seed_papers=seed_papers,
            total_nodes=len(tree.all_nodes),
            total_papers_tracked=sum(n.num_papers for n in tree.all_nodes),
            num_branches=sum(1 for n in tree.all_nodes if n.is_branch_point),
            max_depth=max(len(tree.get_path_to_root(n)) for n in tree.all_nodes),
            active_branches=[
                BranchInfo(
                    branch_id=n.node_id,
                    description=n.description,
                    num_papers=n.num_papers,
                    last_update=n.date_range[1],
                    sample_papers=n.papers[:5]
                )
                for n in tree.get_all_leaf_nodes()
            ],
            date_range=(
                tree.root.date_range[0].isoformat(),
                max(n.date_range[1] for n in tree.all_nodes).isoformat()
            ),
            window_months=request.window_months,
            similarity_threshold=request.similarity_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Paper Detail Endpoint**:
```python
@app.get("/api/paper/{arxiv_id}")
async def get_paper(arxiv_id: str) -> Paper:
    """Get detailed information for single paper"""
    try:
        paper = arxiv_client.get_paper_by_id(arxiv_id)
        return paper
    except Exception as e:
        raise HTTPException(status_code=404, detail="Paper not found")
```

## Frontend Design

### Page Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>ArXiv Concept Evolution Tracker</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="styles.css" rel="stylesheet">
</head>
<body>
    <!-- Header -->
    <header>
        <h1>ArXiv Concept Evolution Tracker</h1>
        <p>Track how scientific concepts evolve and branch over time</p>
    </header>
    
    <!-- Main Content -->
    <main>
        <!-- Step 1: Seed Selection -->
        <section id="seed-selection" class="active">
            <h2>Step 1: Find Seed Papers</h2>
            <div class="search-box">
                <input type="text" id="search-query" placeholder="Search ArXiv...">
                <button onclick="searchPapers()">Search</button>
            </div>
            <div id="search-results"></div>
            <div id="selected-seeds"></div>
            <button onclick="startTracking()" id="track-button" disabled>
                Track Concept Evolution
            </button>
        </section>
        
        <!-- Step 2: Tree Display -->
        <section id="tree-display" style="display: none;">
            <h2>Concept Evolution Tree</h2>
            <div id="tree-visualization"></div>
            <div id="node-details"></div>
            <div id="branch-legend"></div>
        </section>
    </main>
    
    <script src="app.js"></script>
</body>
</html>
```

### Key UI Components

**Search Interface**:
- Text input for keyword search
- Search button with loading state
- Results displayed as cards (title, authors, date, abstract preview)
- Checkbox to select as seed (max 5)

**Selected Seeds Display**:
- List of selected papers
- Remove button for each
- "Track" button enabled when 1-5 seeds selected

**Tree Visualization** (D3.js):
```javascript
function renderTree(treeData) {
    // Use D3.js tree layout
    const width = 1200;
    const height = 800;
    
    const svg = d3.select("#tree-visualization")
        .append("svg")
        .attr("width", width)
        .attr("height", height);
    
    const treeLayout = d3.tree()
        .size([height - 100, width - 200]);
    
    // Convert tree data to D3 hierarchy
    const root = d3.hierarchy(treeData, d => d.children);
    treeLayout(root);
    
    // Draw links (edges)
    svg.selectAll('.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkHorizontal()
            .x(d => d.y)
            .y(d => d.x)
        )
        .style('stroke', '#999')
        .style('stroke-width', 2);
    
    // Draw nodes
    const nodes = svg.selectAll('.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.y},${d.x})`)
        .on('click', (event, d) => showNodeDetails(d.data));
    
    // Node circles
    nodes.append('circle')
        .attr('r', d => d.data.is_branch_point ? 10 : 7)
        .style('fill', d => d.data.is_branch_point ? '#ff6b6b' : '#4ecdc4')
        .style('stroke', '#333')
        .style('stroke-width', 2);
    
    // Node labels
    nodes.append('text')
        .attr('dy', -15)
        .attr('text-anchor', 'middle')
        .text(d => `${d.data.num_papers} papers`)
        .style('font-size', '12px');
    
    // Date labels
    nodes.append('text')
        .attr('dy', 25)
        .attr('text-anchor', 'middle')
        .text(d => formatDate(d.data.date_range[0]))
        .style('font-size', '10px')
        .style('fill', '#666');
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
}
```

**Node Details Panel**:
- Shows when user clicks a node
- Node description
- Date range
- Number of papers
- Confidence score
- Top 5-10 papers from this node:
  - Title (linked to ArXiv)
  - Authors
  - Similarity score
  - Abstract (expandable)
- Path to root visualization
- Children branches (if any)

**Branch Legend**:
- Shows active branches (leaf nodes)
- Color-coded by branch
- Summary statistics per branch
- Click to highlight branch in tree

### JavaScript API Client

```javascript
class ApiClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async searchPapers(query, limit = 20) {
        const response = await fetch(
            `${this.baseUrl}/api/search?query=${encodeURIComponent(query)}&limit=${limit}`
        );
        return response.json();
    }
    
    async trackConcept(seedIds, endDate, windowMonths = 6) {
        const response = await fetch(`${this.baseUrl}/api/track`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                seed_paper_ids: seedIds,
                end_date: endDate,
                window_months: windowMonths
            })
        });
        return response.json();
    }
    
    async getPaper(arxivId) {
        const response = await fetch(`${this.baseUrl}/api/paper/${arxivId}`);
        return response.json();
    }
}
```

## Technology Stack

### Backend
- **Python**: 3.10+
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **NumPy**: Array operations
- **scikit-learn**: Cosine similarity, HDBSCAN clustering
- **arxiv**: ArXiv API client
- **voyageai** or **sentence-transformers**: Embeddings
- **python-dotenv**: Configuration
- **requests**: HTTP client (optional: Semantic Scholar API)

### Frontend
- **Vanilla JavaScript**: No framework for MVP simplicity
- **D3.js**: Tree visualization
- **Plotly.js**: Alternative visualization (optional)
- **CSS**: Custom styling (or Tailwind)

### Development Tools
- **pytest**: Testing
- **black**: Code formatting
- **mypy**: Type checking
- **ruff**: Linting

## Project Structure

```
arxiv-concept-tracker/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration and environment
│   ├── models.py            # Pydantic models (Paper, Node, Tree, etc.)
│   ├── arxiv_client.py      # ArXiv API wrapper
│   ├── embedding_service.py # Embedding generation
│   ├── kalman_tracker.py    # KalmanConceptTracker implementation
│   ├── concept_tree.py      # ConceptTree and ConceptNode
│   ├── api/
│   │   ├── __init__.py
│   │   ├── search.py        # Search endpoint
│   │   ├── track.py         # Tracking endpoint
│   │   └── tree.py          # Tree export/query endpoints
│   └── utils/
│       ├── __init__.py
│       ├── cache.py         # Embedding cache
│       ├── clustering.py    # Branch detection (HDBSCAN)
│       └── similarity.py    # Similarity calculations
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js               # Tree visualization with D3.js
├── tests/
│   ├── test_arxiv_client.py
│   ├── test_kalman_tracker.py
│   ├── test_concept_tree.py
│   └── test_api.py
├── cache/                   # Embedding cache directory
├── .env.example            # Environment template
├── requirements.txt
├── pyproject.toml          # Python project config
└── README.md
```

## Implementation Phases

### Phase 1: Backend Core (Days 1-4)

**Day 1: Project Setup & ArXiv Client**
- [ ] Set up project structure
- [ ] Install dependencies (including HDBSCAN)
- [ ] Implement ArXivClient class
- [ ] Test fetching papers by ID and search
- [ ] Handle rate limiting

**Day 2: Embedding Service**
- [ ] Implement EmbeddingService class
- [ ] Set up Voyage API or Qwen3 local
- [ ] Implement caching mechanism
- [ ] Test embedding generation
- [ ] Measure performance

**Day 3: Kalman Tracker**
- [ ] Implement KalmanConceptTracker class
- [ ] Test velocity and acceleration constraints
- [ ] Implement multi-signal paper evaluation
- [ ] Validate with synthetic data (test jumps are rejected)
- [ ] Tune threshold parameters

**Day 4: Concept Tree**
- [ ] Implement ConceptNode and ConceptTree classes
- [ ] Test branch detection with HDBSCAN
- [ ] Implement tree growth algorithm
- [ ] Test with mock data (verify branching works)
- [ ] Implement tree serialization

### Phase 2: Integration & API (Days 5-7)

**Day 5: Integrated Tracking Algorithm**
- [ ] Combine KalmanTracker + ConceptTree
- [ ] Implement main tracking loop
- [ ] Test with "transformer" papers 2017-2024
- [ ] Verify branch detection (should find BERT/GPT split)
- [ ] Tune parameters (velocity limits, cluster size)

**Day 6: FastAPI Endpoints**
- [ ] Set up FastAPI app
- [ ] Implement search endpoint
- [ ] Implement track endpoint (returns tree)
- [ ] Implement tree query endpoints
- [ ] Add CORS middleware
- [ ] Test with curl/Postman

**Day 7: Error Handling & Validation**
- [ ] Add request validation
- [ ] Implement error handlers
- [ ] Add logging (especially for Kalman rejections)
- [ ] Write API tests
- [ ] Document API with OpenAPI

### Phase 3: Frontend (Days 8-11)

**Day 8: Basic UI Structure**
- [ ] Create HTML layout
- [ ] Add CSS styling
- [ ] Implement search interface
- [ ] Implement seed selection
- [ ] Test API integration

**Day 9: Tree Visualization (D3.js)**
- [ ] Set up D3.js tree layout
- [ ] Render nodes and edges
- [ ] Add node styling (branch points different)
- [ ] Implement click handlers
- [ ] Test with sample tree data

**Day 10: Interactive Features**
- [ ] Node details panel
- [ ] Paper cards with metadata
- [ ] Branch legend/summary
- [ ] Path highlighting
- [ ] Expand/collapse branches

**Day 11: Polish & UX**
- [ ] Add loading states
- [ ] Improve error messages
- [ ] Responsive layout
- [ ] Add examples/documentation
- [ ] Tooltips and help text

### Phase 4: Testing & Validation (Days 12-14)

**Day 12: Integration Testing**
- [ ] End-to-end test: search → select → track
- [ ] Test with transformers (verify BERT/GPT branches)
- [ ] Test with diffusion models
- [ ] Test Kalman filter rejections
- [ ] Performance testing

**Day 13: Validation & Tuning**
- [ ] Validate branch detection accuracy
- [ ] Tune velocity/acceleration limits
- [ ] Tune clustering parameters
- [ ] Test with multiple concepts
- [ ] Fix bugs

**Day 14: Documentation & Deploy**
- [ ] Write README with setup instructions
- [ ] Document API endpoints
- [ ] Add usage examples
- [ ] Create demo video/screenshots
- [ ] Deploy locally for demo
- [ ] Gather feedback from ML researchers

## Testing Strategy

### Unit Tests

**ArXivClient Tests**:
```python
def test_search_papers():
    client = ArXivClient()
    papers = client.search_papers("attention", max_results=5)
    assert len(papers) == 5
    assert all(isinstance(p, Paper) for p in papers)

def test_get_paper_by_id():
    client = ArXivClient()
    paper = client.get_paper_by_id("1706.03762")  # Attention Is All You Need
    assert paper.title.lower().find("attention") >= 0
```

**KalmanConceptTracker Tests**:
```python
def test_velocity_constraint():
    tracker = KalmanConceptTracker(max_velocity=0.05)
    tracker.initialize([get_test_paper("1706.03762")])
    
    # Create paper with impossible velocity
    far_vector = tracker.position + 0.1 * np.random.randn(1024)
    far_vector /= np.linalg.norm(far_vector)
    
    is_valid, confidence, reason = tracker.evaluate_candidate(far_vector)
    assert not is_valid
    assert "velocity" in reason.lower()

def test_acceleration_constraint():
    tracker = KalmanConceptTracker(max_acceleration=0.02)
    tracker.initialize([get_test_paper("1706.03762")])
    
    # Set velocity
    tracker.velocity = 0.03 * np.random.randn(1024)
    
    # Create paper requiring high acceleration
    rapid_change = tracker.position + 0.05 * np.random.randn(1024)
    rapid_change /= np.linalg.norm(rapid_change)
    
    is_valid, confidence, reason = tracker.evaluate_candidate(rapid_change)
    assert not is_valid
    assert "acceleration" in reason.lower()

def test_high_similarity_auto_include():
    tracker = KalmanConceptTracker()
    tracker.initialize([get_test_paper("1706.03762")])
    
    # Very similar paper
    similar = tracker.position + 0.01 * np.random.randn(1024)
    similar /= np.linalg.norm(similar)
    
    is_valid, confidence, reason = tracker.evaluate_candidate(similar)
    assert is_valid
    assert confidence > 0.9
```

**ConceptTree Tests**:
```python
def test_tree_initialization():
    papers = [get_test_paper("1706.03762")]
    tree = ConceptTree(papers)
    
    assert tree.root is not None
    assert tree.root.num_papers == 1
    assert len(tree.root.children) == 0

def test_branch_detection():
    tree = ConceptTree([get_test_paper("1706.03762")])
    
    # Create two distinct clusters of papers
    cluster1 = create_similar_papers(tree.root.concept_vector, n=10, noise=0.1)
    cluster2 = create_similar_papers(tree.root.concept_vector + 0.3, n=10, noise=0.1)
    
    all_papers = cluster1 + cluster2
    branches = tree.detect_branches(all_papers, min_cluster_size=5)
    
    assert branches is not None
    assert len(branches) == 2

def test_tree_growth():
    papers = [get_test_paper("1706.03762")]
    tree = ConceptTree(papers)
    
    new_papers = [get_test_paper("1810.04805")]  # BERT
    children = tree.grow_branch(tree.root, new_papers, datetime(2019, 1, 1))
    
    assert len(children) == 1
    assert tree.root.children[0] == children[0]
    assert children[0].parent == tree.root
```

### Integration Tests

**API Tests**:
```python
from fastapi.testclient import TestClient

def test_track_endpoint_with_branching():
    request = {
        "seed_paper_ids": ["1706.03762"],
        "end_date": "2021-01-01",
        "window_months": 6
    }
    response = client.post("/api/track", json=request)
    assert response.status_code == 200
    
    data = response.json()
    assert "tree_root" in data
    assert "active_branches" in data
    assert data["num_branches"] >= 0  # May or may not branch
    
    # Verify tree structure
    root = data["tree_root"]
    assert root["node_id"] == "root"
    assert isinstance(root["children"], list)
```

### Manual Validation

Test with known concept evolutions:

1. **Transformers** (2017-2024):
   - Should find original Vaswani paper
   - Should detect branch around 2018-2019
   - Branch 1: BERT, encoder-only models
   - Branch 2: GPT series, decoder-only models
   - Possibly Branch 3: Vision transformers (2020)
   - Validate Kalman filter prevents drift to unrelated NLP

2. **Diffusion Models** (2015-2024):
   - Should track from early denoising work
   - Branch into: image generation, video, audio
   - Should find DDPM, DDIM, Stable Diffusion
   - Validate doesn't jump to unrelated generative models (GANs, VAEs)

3. **GANs** (2014-2024):
   - Original GAN paper
   - Branch detection: DCGAN, StyleGAN, conditional GANs
   - Validate velocity constraints during rapid innovation period (2016-2018)

## Configuration

### Environment Variables

```bash
# .env file
VOYAGE_API_KEY=voyage_xxxxx        # For Voyage embeddings
EMBEDDING_MODEL=voyage-3-lite      # or "qwen3-local"
CACHE_DIR=./cache                  # Embedding cache location
LOG_LEVEL=INFO                     # Logging level
ARXIV_RATE_LIMIT=3.0              # Seconds between requests
```

### Configurable Parameters

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Embedding
    embedding_model: str = "voyage-3-lite"
    voyage_api_key: str | None = None
    cache_dir: str = "./cache"
    
    # ArXiv
    arxiv_rate_limit: float = 3.0
    arxiv_max_retries: int = 3
    
    # Tracking - Kalman Filter
    default_window_months: int = 6
    max_velocity: float = 0.05          # Max concept drift per time step
    max_acceleration: float = 0.02      # Max change in velocity
    process_noise: float = 0.01         # Natural drift
    measurement_noise: float = 0.1      # Embedding uncertainty
    
    # Tracking - Thresholds
    threshold_auto_include: float = 0.85    # Auto-accept
    threshold_strong: float = 0.75          # Accept if velocity OK
    threshold_moderate: float = 0.65        # Ambiguous
    threshold_reject: float = 0.55          # Reject
    
    # Branching
    min_cluster_size: int = 5               # Min papers per branch
    min_samples: int = 3                    # HDBSCAN parameter
    max_papers_per_window: int = 500
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## Performance Considerations

### Bottlenecks

1. **Embedding generation**: ~100ms per paper (API) or ~50ms (local)
2. **ArXiv API**: Rate limited to 1 req/3 seconds
3. **Vector similarity**: O(n*d) where n=papers, d=dimensions

### Optimizations

**Caching**:
- Cache all embeddings permanently
- Cache ArXiv metadata for 24 hours
- LRU cache for similarity calculations

**Batch Processing**:
- Embed papers in batches of 100
- Fetch ArXiv papers in batches

**Parallel Processing** (future):
- Parallelize embedding generation
- Use multiprocessing for similarity calculations

### Estimated Performance

For tracking "transformers" 2017-2024:
- ~5000 papers to evaluate
- Voyage API: 5000 * 100ms = 8.3 minutes (one-time, then cached)
- Kalman evaluation: ~1ms per paper = 5 seconds
- HDBSCAN clustering: ~2 seconds per window with branches
- Tree construction: <1 second
- Total first run: ~10 minutes
- Total cached run: ~3 minutes (just Kalman + clustering)

Branching overhead:
- Each branch requires separate tracker state
- If concept splits into 3 branches, tracking time increases ~2x
- Memory: negligible (each tracker is just two vectors)

Bottlenecks:
1. **Embedding generation**: Rate-limited by API or local model speed
2. **HDBSCAN clustering**: O(n log n) for n papers, acceptable for n<1000
3. **Tree rendering**: D3.js handles thousands of nodes efficiently

## Security Considerations

### API Security

- Input validation on all endpoints
- Rate limiting to prevent abuse
- API key protection (environment variables)
- No user data storage (stateless)

### Data Privacy

- No personal data collected
- All ArXiv data is public domain
- Embeddings cached locally
- No tracking or analytics

## Deployment

### Local Development

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
# Served by FastAPI static files or separate server
```

### Production Considerations (Future)

- Use gunicorn + uvicorn workers
- PostgreSQL for caching (instead of pickle)
- Redis for session management
- Docker containerization
- Cloud deployment (AWS/GCP/Heroku)

## Future Enhancements

### Version 2 Features

1. **Bidirectional tracking**: Start from peak, track to origin
2. **Branch detection**: Identify when concepts split
3. **Section-level embeddings**: Methods vs Results comparison
4. **Citation integration**: Overlay citation network
5. **Terminology tracking**: Show how terms evolve
6. **Comparison mode**: Track multiple concepts side-by-side

### Version 3 Features

1. **User accounts**: Save tracked concepts
2. **Collaborative features**: Share discoveries
3. **Advanced visualization**: 2D embedding space
4. **Predictive tracking**: Forecast emerging topics
5. **Paper recommendations**: Suggest papers to read
6. **Full-text indexing**: Download and index PDFs

## Success Metrics

### Technical Metrics
- API response time < 2s (search), < 30s (track)
- Test coverage > 80%
- Zero critical bugs in production

### Product Metrics
- Successfully track 5+ known concept evolutions
- Positive feedback from 3+ ML researchers
- Clear, actionable timeline visualizations

## Glossary

See requirements document for comprehensive glossary.
