# MediScan AI

**A multi-agent AI system for chest X-ray triage and structured radiology reporting.**

MediScan AI pipes an uploaded chest X-ray through four coordinated agents — a dedicated computer-vision screening model, an LLM acting as a radiologist, an independent safety-validation layer, and a report formatter — to produce a clinician-style structured report while actively suppressing unconfirmed urgent flags.

🔗 **Live App:** [medi-scan-ai-nine.vercel.app](https://medi-scan-ai-nine.vercel.app/)
🔗 **API:** [nidhish10-mediscanai.hf.space](https://nidhish10-mediscanai.hf.space)

---

## How It Works

```
React Frontend → FastAPI Backend → Orchestrator
                                        │
                        ┌───────────────┼───────────────┬───────────────┐
                        ▼               ▼                ▼               ▼
                 1. Vision Agent  2. Radiologist  3. Safety Agent  4. Formatting
                 (TorchXRayVision)  (Groq LLM)    (Flag validation)  Agent
```

| # | Agent | Role |
|---|-------|------|
| 1 | **Vision Screening Agent** | Runs TorchXRayVision (DenseNet-121) on the image and returns ranked pathology probabilities across 18 classes. |
| 2 | **Radiologist Agent** | A Groq-hosted vision LLM reads the raw image *and* the CV findings, and produces a structured, section-by-section radiology report. |
| 3 | **Safety Validation Agent** | Independently cross-checks every "urgent" flag from the CV model against the LLM's own visual confirmation, and suppresses any flag the LLM did not corroborate. |
| 4 | **Report Formatting Agent** | Assembles the final payload — report, findings, safety summary, and per-agent timing — for the frontend. |

Each agent reports its own status and elapsed time, so the frontend can show live pipeline progress as a request is processed.

---

## Tech Stack

**Backend**
- FastAPI + Uvicorn (async, with a lifespan hook that pre-loads the CV model on startup)
- TorchXRayVision (DenseNet-121) for chest X-ray pathology screening
- Groq API (Llama-4 Scout) as the vision-capable LLM — provider is swappable via an `LLM_PROVIDER` env variable, with a Claude (`claude-sonnet-4-6`) code path already scaffolded in
- Pillow, NumPy, scikit-image for image preprocessing
- Deterministic fallback inference if the CV model fails to load, so the pipeline never breaks

**Frontend**
- React single-page app
- Multipart upload with optional patient metadata (name, age, sex, referring doctor, clinical history)
- Live multi-agent progress UI driven by the backend's agent trace

**Infrastructure**
- Dockerized backend (CPU-only PyTorch build), deployed on Hugging Face Spaces
- Frontend deployed on Vercel
- CORS-enabled REST API with a `/health` check endpoint

---

## API

### `POST /analyze`

Multipart form upload.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image (jpeg/png/webp, ≤15MB) | ✅ | Chest X-ray image |
| `patient_name` | string | – | Patient name |
| `patient_age` | string | – | Patient age |
| `patient_sex` | string | – | Patient sex |
| `ref_doctor` | string | – | Referring doctor |
| `history` | string | – | Clinical history / context |

**Response** includes: ranked CV findings, the structured radiologist report, the safety-validation summary (including any suppressed flags), the full four-agent trace with timings, and image metadata.

### `GET /health`

Returns service status and pipeline version.

---

## Running Locally

### Backend

```bash
git clone <repo-url>
cd mediscan-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env            # add your GROQ_API_KEY

python run.py                   # serves on http://localhost:7860
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker build -t mediscan-ai .
docker run -p 7860:7860 --env-file .env mediscan-ai
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq (used by the Radiologist Agent) |

---

## Safety Design

The Safety Validation Agent exists because a CV model and an LLM can each be confidently wrong in different ways. Rather than trusting either output in isolation, every CV-flagged urgent finding must also be independently named as urgent in the LLM's own report text before it survives to the final output. Unconfirmed flags are logged and suppressed, and the discrepancy is recorded in the response.

The Radiologist Agent's prompt also encodes explicit clinical guardrails — projection-aware reasoning (AP vs. PA), pediatric-specific normals, medical device position checks, and a strict evidentiary bar for flagging anything urgent (e.g., pneumothorax requires a visible pleural line, not just a hyperlucent area).

---

## Disclaimer

MediScan AI is a hackathon prototype for educational and demonstration purposes. It is **not** a certified medical device and must not be used for actual clinical diagnosis or patient care decisions.

---

## Roadmap

- [ ] Swap default LLM provider to Claude (`claude-sonnet-4-6`) for stronger reasoning
- [ ] PDF export of the final clinical report
- [ ] Extend the vision agent beyond chest X-ray to additional imaging modalities