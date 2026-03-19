# DualR — Architecture

## Current Runtime

Single EC2 instance (`us-east-2`, stage account) running two Docker containers:

```
Browser
  └─> nginx (port 80) — React SPA
        └─> FastAPI (port 8000) — /api/predict
              ├─> joblib bundles (XGBoost pipeline, feature names)
              ├─> drug probability parquets (P(disease|drug))
              └─> MSU CatChat (optional, novel drugs only)
```

- The frontend serves a static React/Vite build via nginx.
- All scoring requests go to the backend; the browser never touches model files.
- No secrets are passed to or stored by the frontend.

## Source of Truth

GitHub repo `DualR_app` is the single source of truth. Deployment is driven by GitHub Actions with a self-hosted runner on the stage EC2 instance.

## Backend

- **Framework:** FastAPI + uvicorn (Python 3.12)
- **Inference:** Frozen AoU joblib bundles (`deploy_{disease}.joblib`) loaded at startup from `backend/models/`. Each bundle contains a sklearn Pipeline (with XGBoost) and `feature_names`.
- **Drug signal:** Pre-computed probability tables (`drug_probs_{disease}_{mode}.parquet`) encode P(disease|drug) for nocot and cot scoring modes. Six files total (t2d/htn/aud × nocot/cot).
- **Novel drugs:** Drugs absent from the lookup tables are queried via MSU CatChat (`CATCHAT_BASE_URL`). Results are cached in `CACHE_DIR` (JSONL) and reloaded at next startup. Falls back to disease prevalence if CatChat is unconfigured or fails.
- **Three phenotypes:** T2D, Hypertension, AUD

## Frontend

- **Stack:** React 18 + Vite, served as a static build via nginx
- **API:** Calls `POST /api/predict` (proxied by nginx to the backend container). No mock scoring.
- **Privacy:** No cookies, no analytics, no local storage. Session state only.

## File Layout

```
DualR_app/
├── .github/workflows/   CI (ci.yml) and manual deploy (deploy-stage.yml)
├── frontend/            React app, nginx config, Dockerfile
├── backend/             FastAPI app, Dockerfile, requirements
│   ├── app/main.py      Prediction endpoint
│   └── models/          joblib bundles + parquet files (not in git — transfer separately)
├── deploy/              docker-compose.stage.yml, env.stage.example
└── docs/                This file, DEPLOYMENT.md
```

## Out of Scope (for now)

- Multiple environments beyond stage
- CDN, load balancer, container orchestration
- Automated deploys on push (all deploys are manual workflow_dispatch)
