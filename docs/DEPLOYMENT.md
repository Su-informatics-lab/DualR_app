# DualR — Deployment

## Stage Environment

- **Instance:** EC2 (`us-east-2`, stage account), IP `18.117.115.31`
- **Access:** AWS Session Manager (`dualr-stage-ec2-ssm-role`) — no SSH keys needed
- **Runtime:** Docker + Docker Compose

## One-Time Setup (already done)

1. EC2 instance launched with SSM role attached.
2. Docker installed.
3. GitHub Actions self-hosted runner installed and registered with label `dualr-stage`.
4. Transfer model files to `~/DualR_app/backend/models/` on the instance:
   - `xgb_t2d.json`, `xgb_htn.json`, `xgb_aud.json`
   - `drug_probs_{t2d,htn,aud}_{nocot,cot}.parquet`
5. Copy `deploy/env.stage.example` to `deploy/env.stage` and fill in values.

## Routine Deploy

Push to `main`, then trigger the `deploy-stage` workflow manually from GitHub Actions (`Run workflow` button). This runs on the self-hosted runner:

```bash
docker compose -f deploy/docker-compose.stage.yml up -d --build
```

No AWS admin rights required for routine deploys.

## Rebuild / Restart

```bash
# On the EC2 instance via Session Manager
cd ~/DualR_app
docker compose -f deploy/docker-compose.stage.yml up -d --build

# Restart without rebuild
docker compose -f deploy/docker-compose.stage.yml restart

# View logs
docker compose -f deploy/docker-compose.stage.yml logs -f
```

## Health Checks

```bash
curl http://localhost/health         # frontend nginx
curl http://localhost:8000/health    # backend FastAPI (returns loaded models)
```

## Environment Variables

Edit `deploy/env.stage` on the EC2 instance. The file is not tracked in git.

Key variables:
- `MODEL_DIR` — path to model files inside the container (default: `models`)
- `VLLM_BASE_URL` — vLLM endpoint for novel drug queries (optional; omit to skip)

After editing, redeploy with `docker compose ... up -d --build`.

## Rollback

```bash
git revert <commit>
git push origin main
# Then trigger deploy-stage workflow
```

## What Requires Admin Rights

- Changing the EC2 instance type or security groups
- Adding or rotating the SSM role
- Registering a new GitHub Actions runner
- Transferring model files to the instance

Routine code deploys do **not** require admin rights.
