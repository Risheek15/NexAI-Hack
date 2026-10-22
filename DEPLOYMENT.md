# Maintenance Copilot — Production Deployment Guide
**NEXAI 2026 Hackathon MVP**

This guide provides instructions to run, containerize, and deploy the Maintenance Copilot architecture.

---

## 1. Architecture Overview

```
┌─────────────────────────────────┐       ┌──────────────────────────────────────────────┐
│  Next.js 14 Frontend (Port 3000)│ ◄───► │  FastAPI Python Backend (Port 8000)          │
│  - Tailwind CSS + Lucide Icons  │       │  - Bounded Gemini Tool Loop (search/get)     │
│  - Source Logbook Inspector     │       │  - Deterministic Evidence Gate               │
│  - Tool Call Trace Drawer       │       │  - Citation & Exact Quote Validator          │
│  - Interactive Glossary         │       │  - Hybrid Retrieval Engine (BM25 + Semantic) │
└─────────────────────────────────┘       └──────────────────────┬───────────────────────┘
                                                                 │
                                          ┌──────────────────────▼───────────────────────┐
                                          │  Canonical Aviation Logbook Store            │
                                          │  - 6,169 Ingested Records (data/processed/)  │
                                          │  - Cached Embeddings & BM25 Index            │
                                          │  - 332 Discovered Glossary Terms             │
                                          └──────────────────────────────────────────────┘
```

---

## 2. Quick Local Start (Development)

### Prerequisites
- Python 3.10 to 3.13
- Node.js v18.0 or later (v24 tested)
- A Google Gemini API Key

### Step 1: Clone & Configure Environment
```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

Sample `.env` file:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOOL_CALLS=5
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
ENVIRONMENT=production
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Step 2: One-Click Startup

**On Windows (PowerShell):**
```powershell
.\start_all.ps1
```

**On Linux / macOS:**
```bash
chmod +x start_all.sh
./start_all.sh
```

- **Frontend UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 3. Automated Test Suite Execution

Run the complete 61-test automated suite across all layers:
```bash
# In virtual environment:
pytest -v tests
```
*Expected result: 61 passed in ~60s.*

---

## 4. Docker Containerization

### Docker Compose Deployment (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GEMINI_MODEL=gemini-3.5-flash-lite
      - MAX_TOOL_CALLS=5
      - BACKEND_HOST=0.0.0.0
      - BACKEND_PORT=8000
      - CORS_ORIGINS=http://localhost:3000,http://localhost
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped
```

### Backend Dockerfile (`Dockerfile.backend`)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY scripts/ scripts/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile (`frontend/Dockerfile`)
```dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app ./
EXPOSE 3000
CMD ["npm", "run", "start"]
```

---

## 5. Security & Airworthiness Safety Controls

1. **No Hardcoded Secrets**: Secrets are isolated strictly in `.env` and excluded from version control via `.gitignore`.
2. **Untrusted Data Isolation**: All retrieved logbook records injected into prompts are wrapped in explicit `<historical_logbook_record>` XML tags to defend against prompt injection.
3. **Prescriptive Warning**: Responses persistently carry descriptive notices distinguishing historical entries from FAA-approved Airworthiness Directives (ADs) or Maintenance Manuals (AMMs).
4. **Offline Resilience**: If Gemini API keys are absent or rate limits are reached, the system falls back to hybrid semantic search without crashing.
