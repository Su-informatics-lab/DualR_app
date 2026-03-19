# DualR

A clinical risk assessment web app that transforms medication history into phenotypic disease risk estimates using pre-computed drug associations from large language models and XGBoost inference.

**Phenotypes:** Type 2 Diabetes (T2D), Hypertension (HTN), Alcohol Use Disorder (AUD)

Su Lab · Biomedical Informatics, Biostatistics & Health Data Science · Indiana University School of Medicine

---

## Repo Layout

```
DualR_app/
├── .github/workflows/
│   ├── ci.yml                 Runs on push/PR — builds frontend + backend
│   └── deploy-stage.yml       Manual deploy to stage EC2 (workflow_dispatch)
├── frontend/
│   ├── src/app.jsx            React SPA
│   ├── public/dr.png          Logo
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── index.html
│   ├── main.jsx
│   ├── package.json
│   └── vite.config.js
├── backend/
│   ├── app/main.py            FastAPI prediction endpoint
│   ├── models/                XGBoost + parquet files (not in git)
│   ├── Dockerfile
│   └── requirements.txt
├── deploy/
│   ├── docker-compose.stage.yml
│   └── env.stage.example
├── docs/
│   ├── DEPLOYMENT.md
│   └── ARCHITECTURE.md
└── .gitignore
```

## Run Locally

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MODEL_DIR=models uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev        # proxies /api/* to localhost:8000
```

The frontend calls `POST /api/predict` on the backend. The backend loads XGBoost models and drug probability tables from `backend/models/` at startup. Place the AoU model artifacts there before running.

## Stage Deployment

Stage runs on a single EC2 instance (`18.117.115.31`, `us-east-2`). Deployment is manual:

1. Push code to `main`
2. Go to **Actions → deploy-stage → Run workflow**

The workflow runs `docker compose ... up -d --build` on the self-hosted runner on the EC2 instance.

See `docs/DEPLOYMENT.md` for setup details and `docs/ARCHITECTURE.md` for the system overview.

## Secrets

- Never commit `deploy/env.stage` (gitignored)
- Model files go in `backend/models/` on the EC2 instance — not in git
- No secrets belong in frontend code or `VITE_*` variables

## Out of Scope (for now)

- Multi-environment deployment
- CDN, load balancer, auto-scaling
- LLM-backed scoring (vLLM endpoint is wired in the backend but not required for stage)
- Automated deploys on push (all deploys are manual for now)

## License

Apache 2.0
