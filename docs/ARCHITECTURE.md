# DualR — Architecture

## Current Runtime

Single EC2 instance (`us-east-2`, stage account) running two Docker containers:

```
Browser
  └─> nginx (port 80) — React SPA
        └─> FastAPI (port 8000) — /api/predict
              └─> XGBoost models + drug probability tables (local files)
```

- The frontend serves a static React/Vite build via nginx.
- All scoring requests go to the backend; the browser never touches model files.
- No secrets are passed to or stored by the frontend.

## Source of Truth

GitHub repo `DualR_app` is the single source of truth. Deployment is driven by GitHub Actions with a self-hosted runner on the stage EC2 instance.

## Backend

- **Framework:** FastAPI + uvicorn
- **Inference:** XGBoost models (`xgb_t2d.json`, `xgb_htn.json`, `xgb_aud.json`) loaded at startup from `backend/models/`
- **Drug signal:** Pre-computed probability tables (`drug_probs_{disease}_{mode}.parquet`) encode P(disease|drug) from All of Us and INPC cohorts
- **Novel drugs:** Optionally queried via a vLLM endpoint (`VLLM_BASE_URL`); falls back to prevalence if unavailable
- **Three phenotypes:** T2D, Hypertension, AUD

## Frontend

- **Stack:** React 18 + Vite, served as a static build via nginx
- **State:** Current scoring is simulated client-side (placeholder). The backend contract (`POST /api/predict`) is already defined — wire up when models are deployed to EC2.
- **Privacy:** No cookies, no analytics, no local storage. Session state only.

## File Layout

```
DualR_app/
├── .github/workflows/   CI (ci.yml) and manual deploy (deploy-stage.yml)
├── frontend/            React app, nginx config, Dockerfile
├── backend/             FastAPI app, Dockerfile, requirements
│   ├── app/main.py      Prediction endpoint
│   └── models/          XGBoost + parquet files (not in git — transfer separately)
├── deploy/              docker-compose.stage.yml, env.stage.example
└── docs/                This file, DEPLOYMENT.md
```

## Out of Scope (for now)

- LLM-backed scoring path (vLLM endpoint is wired but not required for stage)
- Multiple environments beyond stage
- CDN, load balancer, container orchestration
