# ArXiv Concept Tracker - Showcase Plan

**Project Type**: Portfolio/Training Project
**Timeline**: 1-2 days remaining
**Goal**: Deployable demo showing coding capabilities

## Current Status ✅

**Backend (DONE)**:
- ✅ Linear concept tracking with Kalman filtering
- ✅ ArXiv API integration with caching
- ✅ Qwen3 local embeddings
- ✅ FastAPI REST API (3 endpoints)
- ✅ Validated tracking quality (Transformer evolution 2017-2019)
- ✅ Realistic velocity/acceleration constraints
- ✅ PyCharm debugging setup

**What's Missing**:
- ❌ Web frontend visualization
- ❌ Deployment to Render.com

## MVP for Showcase (Next 1-2 Days)

### Day 1: Web Frontend (Priority 1) 🎨

**Goal**: Interactive timeline visualization

**Tasks**:
- [ ] **Setup** (30 min)
  - [ ] Create `frontend/` directory
  - [ ] Choose stack: React or vanilla HTML/CSS/JS (recommend vanilla for speed)
  - [ ] Basic HTML structure

- [ ] **Seed Selection UI** (2 hours)
  - [ ] Search box for arXiv papers
  - [ ] Display search results (title, authors, year)
  - [ ] Checkbox to select 1-5 seed papers
  - [ ] "Track Concept" button with parameters (end date, window size)

- [ ] **Timeline Visualization** (3-4 hours)
  - [ ] Horizontal timeline showing time windows
  - [ ] Each window = card/section with papers
  - [ ] Paper cards show: title, similarity score, abstract (truncated)
  - [ ] Click to expand abstract
  - [ ] Visual indicators for:
    - Average similarity per window
    - Position drift per window
    - Confidence tier distribution (High/Moderate/Low)

- [ ] **Polish** (1-2 hours)
  - [ ] Loading spinner during tracking
  - [ ] Error messages
  - [ ] Responsive layout (mobile-friendly)
  - [ ] Clean, modern CSS (use Tailwind CDN for speed)

**Deliverable**: Working frontend that visualizes concept evolution

---

### Day 2: Deployment + Final Polish (Priority 2) 🚀

**Morning: Render.com Deployment** (2-3 hours)

- [ ] **Containerization**
  - [ ] Write `Dockerfile` (FastAPI + frontend served via static files)
  - [ ] Test Docker build locally
  - [ ] Create `.dockerignore`

- [ ] **Render Configuration**
  - [ ] Create `render.yaml` for infrastructure-as-code
  - [ ] Set up web service (FastAPI)
  - [ ] Configure environment variables
  - [ ] Set up health check endpoint

- [ ] **Deploy & Test**
  - [ ] Push to GitHub
  - [ ] Connect Render to GitHub repo
  - [ ] Deploy and test live
  - [ ] Fix any deployment issues

**Afternoon: Polish & Documentation** (2-3 hours)

- [ ] **Demo Preparation**
  - [ ] Create example tracking run (Transformers 2017-2019)
  - [ ] Screenshot the results
  - [ ] Test with different seed papers

- [ ] **Documentation**
  - [ ] Update README with:
    - [ ] Live demo link
    - [ ] Screenshots
    - [ ] Quick start guide
    - [ ] Architecture diagram (simple)
  - [ ] Record short demo video (optional, 2-3 min)

- [ ] **Code Cleanup**
  - [ ] Remove debug print statements
  - [ ] Add comments to complex parts
  - [ ] Ensure all tests pass

**Deliverable**: Live demo on Render.com + polished README

---

## Technical Decisions

### Frontend Stack: Vanilla JS (Recommended)

**Why not React?**
- Setup overhead (webpack, babel, etc.)
- Overkill for simple visualization
- Slower development for showcase

**Why vanilla JS?**
- Fastest development
- No build step needed
- Easy to deploy (just static files)
- Shows fundamentals

**Architecture**:
```
frontend/
  index.html          # Single page app
  style.css          # Styles (or use Tailwind CDN)
  app.js             # Frontend logic

backend/
  main.py            # Serve both API + static files
```

### Deployment Architecture

```
Render.com
  │
  ├─ Web Service (Docker)
  │   ├─ FastAPI backend (port 8000)
  │   └─ Static files (frontend/)
  │
  └─ Environment Variables
      ├─ MAX_VELOCITY
      ├─ MAX_ACCELERATION
      └─ LOG_LEVEL
```

**No database needed** - all computation is stateless!

---

## Out of Scope (Don't Build)

- ❌ User accounts / authentication
- ❌ Saving tracking runs (persistence)
- ❌ Branch detection (too complex)
- ❌ Multiple embedding models
- ❌ Comparison features
- ❌ Advanced filters
- ❌ Export to CSV/JSON (API already returns JSON)

---

## Success Criteria

**Must Have**:
- ✅ Live demo URL on Render.com
- ✅ Can search and select seed papers
- ✅ Can track concept evolution (Transformer example)
- ✅ Timeline visualization shows paper progression
- ✅ Similarity and position drift visible
- ✅ Mobile-responsive design

**Nice to Have**:
- ⭐ Smooth animations (timeline appears incrementally)
- ⭐ Dark mode toggle
- ⭐ Paper abstract expansion on click
- ⭐ Direct links to arXiv PDFs

**Showcase Value**:
- 🎯 Shows full-stack capability (Python backend + JS frontend)
- 🎯 Demonstrates ML/NLP knowledge (embeddings, Kalman filtering)
- 🎯 Proves deployment skills (Docker, Render.com)
- 🎯 Clean, documented code
- 🎯 Working live demo

---

## Persistence - Skip It?

**You asked if persistence is justified**. For a **showcase**, NO:

**Arguments Against**:
- Adds complexity (SQLite, ORM, migrations)
- Tracking runs are fast (~30-60 sec)
- No "user accounts" = no "my runs"
- Demo can just run on-demand

**Only justification FOR persistence**:
- If you want to show "I can build full CRUD"
- But you already show that with arXiv search + tracking

**Verdict**: SKIP persistence. Use that time for better visualization.

---

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Frontend setup | 30 min | P0 |
| Seed selection UI | 2 hrs | P0 |
| Timeline visualization | 4 hrs | P0 |
| UI polish | 1 hr | P1 |
| Dockerfile | 1 hr | P0 |
| Render deployment | 2 hrs | P0 |
| Testing & fixes | 2 hrs | P1 |
| Documentation | 1 hr | P1 |
| **Total** | **13.5 hrs** | **~1.5 days** |

With 10x speed from AI assistance = feasible in 1-2 days!

---

## Next Steps

**Ready to start?**

1. **Choose frontend approach**: Vanilla JS or React?
2. **Create frontend directory structure**
3. **Build seed selection interface first** (gives immediate visual feedback)
4. **Then build timeline visualization**
5. **Deploy early, deploy often** (test on Render ASAP)

---

**Questions Before Starting?**

- Frontend framework preference?
- Any specific design inspiration (show me a screenshot)?
- Hosting budget constraints? (Render free tier OK?)
