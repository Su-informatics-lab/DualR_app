"""
DualR Backend — FastAPI
Prediction endpoint using frozen AoU joblib bundles and DualR drug lookup tables.

Feature schema (must match training in ml.py exactly):
- age: numeric integer [18, 120] — continuous feature
- race, ethnicity, gender: categorical-encoded integers (see maps below)
- Charlson comorbidities: binary, all included; pipeline selects via feature_names
- dualr_no_cot, dualr_cot: continuous DualR scores from drug lookup parquets

Inference path:
  1. Load deploy_{disease}.joblib at startup → bundle["pipeline"] + bundle["feature_names"]
  2. Compute DualR scores from parquet lookup tables
  3. Drugs absent from lookup tables → query MSU CatChat (backend-only, no secret in frontend)
  4. Build one-row DataFrame in bundle's feature order → predict_proba
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# Feature Definitions
# ═══════════════════════════════════════════

CHARLSON_COMORBIDITIES = [
    "HIV", "AIDS", "Cerebrovascular_Disease", "Congestive_Heart_Failure",
    "Myocardial_Infarction", "Peripheral_Vascular_Disease",
    "Chronic_Pulmonary_Disease", "Dementia", "Liver_Disease_Mild",
    "Liver_Disease_Moderate_Severe", "Malignancy", "Metastatic_Solid_Tumor",
    "Peptic_Ulcer_Disease", "Renal_Disease_Mild_Moderate",
    "Renal_Disease_Severe", "Rheumatic_Disease", "Hemiplegia_Paraplegia",
    "Diabetes_with_Chronic_Complications", "Diabetes_without_Chronic_Complications",
]

PREVALENCES = {"t2d": 0.109, "htn": 0.330, "aud": 0.078}

# Per-drug log2 OR is clipped to this range, as in dualr_post.py compute_log_odds.
LOG_OR_CLIP = 10.0

# Categorical encoding (matches ml.py reference encoding). ml.py codes the reference as 0
# and the remaining categories in order of first appearance in the AoU EMR file, which
# gave Woman before Others and Others before Black. The bundles' scaler stats confirm
# this (race mean 0.639, sd 0.784 for the AoU race mix).
GENDER_MAP = {"Man": 0, "Woman": 1, "Other": 2}
RACE_MAP = {"White": 0, "Others": 1, "Black": 2}
ETHNICITY_MAP = {"Others": 0, "Hispanic": 1}

MODEL_DIR = os.getenv("MODEL_DIR", "models")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/dualr_cache")

# CatChat (MSU) — backend-only fallback for novel drugs
CATCHAT_BASE_URL = os.getenv("CATCHAT_BASE_URL", "")
CATCHAT_MODEL = os.getenv("CATCHAT_MODEL", "")
CATCHAT_API_KEY = os.getenv("CATCHAT_API_KEY", "")
# Per-call timeout. gpt-oss reasons before answering, so a CoT call can take minutes.
CATCHAT_TIMEOUT = float(os.getenv("CATCHAT_TIMEOUT", "180"))
# Total CatChat wall-clock budget per request. Background jobs (/api/predict/jobs)
# report progress, so they can wait; the synchronous /api/predict must answer within
# nginx's 60 s proxy timeout. Calls still running at the budget are cancelled.
CATCHAT_JOB_BUDGET = float(os.getenv("CATCHAT_JOB_BUDGET", "600"))
CATCHAT_SYNC_BUDGET = float(os.getenv("CATCHAT_SYNC_BUDGET", "45"))
# Matches DEFAULT_MAX_TOKENS in the research code (dualr_oss.py); reasoning tokens
# count toward this limit, so small values leave the final answer empty.
CATCHAT_MAX_TOKENS = int(os.getenv("CATCHAT_MAX_TOKENS", "4096"))
CATCHAT_CONCURRENCY = int(os.getenv("CATCHAT_CONCURRENCY", "12"))

# Finished jobs (and their results) are kept in memory this long, then dropped.
JOB_TTL = float(os.getenv("JOB_TTL", "600"))

# ═══════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════

bundles: dict = {}      # disease -> {"pipeline": ..., "feature_names": [...]}
drug_probs: dict = {}   # disease -> {drug_name -> {"nocot": p, "cot": p}}


def _load_runtime_cache():
    """Merge previously cached novel drug probabilities into drug_probs."""
    cache_path = Path(CACHE_DIR)
    if not cache_path.exists():
        return
    for disease in ["t2d", "htn", "aud"]:
        for mode in ["nocot", "cot"]:
            fpath = cache_path / f"{disease}_{mode}.jsonl"
            if not fpath.exists():
                continue
            count = 0
            with open(fpath) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        drug = entry["drug"]
                        p = float(entry["probability"])
                        if drug not in drug_probs[disease]:
                            drug_probs[disease][drug] = {}
                        if mode not in drug_probs[disease][drug]:
                            drug_probs[disease][drug][mode] = p
                            count += 1
                    except (KeyError, ValueError, json.JSONDecodeError) as e:
                        logger.error(
                            f"Corrupt cache entry in {fpath}, line: {line.strip()[:100]}; error: {e}"
                        )
                        continue
            if count:
                logger.info(f"Loaded {count} cached novel drug probs from {fpath}")


def load_models():
    """Load joblib bundles and drug probability lookup tables at startup."""
    for disease in ["t2d", "htn", "aud"]:
        bundle_path = os.path.join(MODEL_DIR, f"deploy_{disease}.joblib")
        if os.path.exists(bundle_path):
            bundle = joblib.load(bundle_path)
            bundles[disease] = bundle
            feat_count = len(bundle.get("features", []))
            logger.info(f"Loaded bundle: {bundle_path} ({feat_count} features)")
        else:
            logger.warning(f"Bundle not found: {bundle_path}")

        drug_probs[disease] = {}
        for mode in ["nocot", "cot"]:
            prob_path = os.path.join(MODEL_DIR, f"drug_probs_{disease}_{mode}.parquet")
            if os.path.exists(prob_path):
                df = pd.read_parquet(prob_path)
                drug_col = "drug" if "drug" in df.columns else "standard_concept_name"
                for _, row in df.iterrows():
                    drug_name = str(row[drug_col]).strip()
                    if drug_name not in drug_probs[disease]:
                        drug_probs[disease][drug_name] = {}
                    drug_probs[disease][drug_name][mode] = float(row["probability"])
                logger.info(f"Loaded {len(df)} drug probs: {prob_path}")
            else:
                logger.warning(f"Drug probs not found: {prob_path}")

    _load_runtime_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="DualR API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════
# DualR Score Computation
# ═══════════════════════════════════════════

def compute_dualr_score(
    drug_names: list[str],
    disease: str,
    mode: str,  # "nocot" or "cot"
    baseline_prob: float,
) -> float:
    """
    Sum of per-drug log2 odds ratios (each clipped to +/-LOG_OR_CLIP) for known drugs
    relative to disease prevalence. Matches dualr_post.py aggregation.
    """
    probs_table = drug_probs.get(disease, {})
    log_odds = []
    for drug in drug_names:
        entry = probs_table.get(drug.strip(), {})
        if mode in entry:
            p = max(1e-10, min(1 - 1e-10, entry[mode]))
            drug_odds = p / (1 - p)
            base_odds = baseline_prob / (1 - baseline_prob)
            or_val = max(1e-10, drug_odds / base_odds)
            log_odds.append(float(np.clip(np.log2(or_val), -LOG_OR_CLIP, LOG_OR_CLIP)))
    return sum(log_odds) if log_odds else 0.0


def _write_cache(drug: str, disease: str, mode: str, probability: float):
    """Append a novel drug probability to the runtime cache (best-effort)."""
    try:
        cache_path = Path(CACHE_DIR)
        cache_path.mkdir(parents=True, exist_ok=True)
        fpath = cache_path / f"{disease}_{mode}.jsonl"
        with open(fpath, "a") as f:
            f.write(json.dumps({"drug": drug, "probability": probability}) + "\n")
    except Exception as e:
        logger.error(f"Runtime cache write failed for {drug}/{disease}/{mode}: {e}")
        raise


async def query_catchat(
    drug_name: str, disease: str, use_cot: bool, client: httpx.AsyncClient
) -> float | None:
    """
    Query MSU CatChat for P(disease|drug) for a drug absent from the lookup tables.
    Reads CATCHAT_BASE_URL, CATCHAT_MODEL, CATCHAT_API_KEY from environment.
    Raises RuntimeError if CatChat is unconfigured (hard infrastructure failure).
    Returns None if the request fails or response contains no parseable probability;
    the drug is then skipped (contributes 0 to the DualR score), matching dualr_post.py dropna.
    """
    if not CATCHAT_BASE_URL or not CATCHAT_MODEL:
        raise RuntimeError(
            f"CatChat not configured (CATCHAT_BASE_URL={CATCHAT_BASE_URL!r}, "
            f"CATCHAT_MODEL={CATCHAT_MODEL!r}); cannot score novel drug: {drug_name}"
        )

    disease_names = {
        "t2d": "type 2 diabetes",
        "htn": "hypertension",
        "aud": "alcohol use disorder",
    }
    disease_full = disease_names.get(disease, disease)

    if use_cot:
        prompt = (
            f"Given that a patient was prescribed {drug_name}, estimate the probability "
            f"that they have {disease_full}. Think step by step, then give your final "
            f"answer as a single decimal number between 0 and 1."
        )
    else:
        prompt = (
            f"Given that a patient was prescribed {drug_name}, estimate the probability "
            f"that they have {disease_full}. Respond with only a decimal number between 0 and 1."
        )

    headers = {"Content-Type": "application/json"}
    if CATCHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CATCHAT_API_KEY}"

    try:
        resp = await client.post(
            f"{CATCHAT_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": CATCHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": CATCHAT_MAX_TOKENS,
                "temperature": 0.01,
                **( {"reasoning_effort": "medium"} if "oss" in CATCHAT_MODEL.lower() else {} ),
            },
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        numbers = re.findall(r"0\.\d+", text)
        if numbers:
            return float(numbers[-1])
        logger.warning(
            f"CatChat returned no parseable probability for drug={drug_name}, "
            f"disease={disease}; skipping. finish_reason={choice.get('finish_reason')}, "
            f"Raw: {text[:200]}"
        )
        return None
    except Exception as e:
        logger.error(f"CatChat query failed for drug={drug_name}, disease={disease}: {e}")
        return None


def predict_risk(bundle: dict, rows: pd.DataFrame) -> float:
    """
    Probability of the disease for one feature row, on the cohort's prevalence scale.

    The AoU models were trained with scale_pos_weight = N_neg / N_pos (equal to
    (1 - prevalence) / prevalence, as the full cohort was used), which inflates
    predict_proba toward 0.5. Undo the class weighting with the prior-shift
    correction p / (p + (1 - p) * w), w = scale_pos_weight. This removes the
    reweighting only; it is not a fitted calibration, and ranking (AUC) is unchanged.
    """
    pipe = bundle["pipeline"]
    p = float(pipe.predict_proba(rows)[0][1])
    w = pipe[-1].get_params().get("scale_pos_weight") or 1.0
    return p / (p + (1.0 - p) * w)


@dataclass
class Progress:
    """Work units for one prediction: the table lookup, one per CatChat call, one per disease model."""
    total: int = 0
    done: int = 0
    stage: str = "queued"  # queued -> estimating -> modeling -> done
    drugs: dict[str, list[int]] = field(default_factory=dict)  # drug -> [done, total] calls

    def tick(self, drug: str | None = None):
        self.done = min(self.done + 1, self.total)
        if drug in self.drugs:
            self.drugs[drug][0] += 1

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "done": self.done,
            "stage": self.stage,
            "drugs": [{"name": n, "done": d, "total": t} for n, (d, t) in self.drugs.items()],
        }


async def score_novel_drugs(
    novel: dict[str, list[str]], budget: float, progress: Progress
) -> dict[str, list[str]]:
    """
    Query CatChat concurrently for every (disease, drug, mode) absent from the lookup
    tables and add the results to drug_probs. Calls still running after `budget`
    seconds are cancelled; their drugs are skipped like any unparseable response.
    Each finished or cancelled call advances `progress`.
    Returns disease -> drugs that received no probability in either mode.
    """
    jobs = [
        (disease, drug, use_cot, mode)
        for disease, drugs in novel.items()
        for drug in drugs
        for use_cot, mode in [(False, "nocot"), (True, "cot")]
    ]
    if not jobs:
        return {disease: [] for disease in novel}

    sem = asyncio.Semaphore(CATCHAT_CONCURRENCY)

    async with httpx.AsyncClient(timeout=CATCHAT_TIMEOUT) as client:
        async def run(job):
            disease, drug, use_cot, mode = job
            async with sem:
                t0 = time.monotonic()
                p = await query_catchat(drug, disease, use_cot, client)
                logger.info(
                    f"CatChat {mode} {disease} {drug!r}: {time.monotonic() - t0:.1f}s, "
                    f"{'probability' if p is not None else 'no probability'}"
                )
            progress.tick(drug)
            return job, p

        task_jobs = {asyncio.create_task(run(job)): job for job in jobs}
        tasks = list(task_jobs)
        try:
            done, pending = await asyncio.wait(tasks, timeout=budget)
        finally:
            # On budget expiry, and also when the whole job is cancelled, stop the calls.
            unfinished = [t for t in tasks if not t.done()]
            for task in unfinished:
                task.cancel()
            await asyncio.gather(*unfinished, return_exceptions=True)
        if pending:
            logger.warning(f"CatChat budget of {budget:.0f}s exceeded; cancelled {len(pending)} of {len(jobs)} calls")
            for task in pending:
                progress.tick(task_jobs[task][1])

    scored = set()
    for task in done:
        (disease, drug, _, mode), p = task.result()
        if p is not None:
            drug_probs[disease].setdefault(drug, {})[mode] = p
            _write_cache(drug, disease, mode, p)
            scored.add((disease, drug))
    return {
        disease: [drug for drug in drugs if (disease, drug) not in scored]
        for disease, drugs in novel.items()
    }


# ═══════════════════════════════════════════
# API Models
# ═══════════════════════════════════════════

class PredictRequest(BaseModel):
    diseases: list[str]     # e.g., ["t2d", "htn"]
    demographics: dict      # {"age": 45, "gender": "Man", "race": "White", "ethnicity": "Others"}
    comorbidities: dict     # {"HIV": 0, "Dementia": 1, ...}
    drugs: list[str]        # ["metformin hydrochloride 500 MG...", ...]

class PredictResponse(BaseModel):
    results: dict           # disease -> {risk, dualr_nocot, dualr_cot, top_drugs}


# ═══════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(bundles.keys())}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "models_loaded": list(bundles.keys())}


def validate(req: PredictRequest) -> None:
    """Reject bad input before any work starts (also before a background job is created)."""
    for disease in req.diseases:
        if disease not in PREVALENCES:
            raise HTTPException(400, f"Unknown disease: {disease}")
        if disease not in bundles:
            raise HTTPException(503, f"Model not loaded for disease: {disease}")
    try:
        age_val = int(req.demographics.get("age"))
    except (TypeError, ValueError):
        raise HTTPException(400, "demographics.age must be an integer")
    if not (18 <= age_val <= 120):
        raise HTTPException(400, f"demographics.age must be 18-120, got {age_val}")


async def run_prediction(req: PredictRequest, budget: float, progress: Progress) -> dict:
    """Score novel drugs with CatChat (within `budget` seconds), then run each disease model."""
    results = {}

    # Drugs absent from each disease's lookup table (capped at 10 per disease) are scored
    # with CatChat as typed, for all diseases at once before the per-disease loop.
    novel_by_disease = {
        disease: [d for d in req.drugs if d.strip() not in drug_probs.get(disease, {})]
        for disease in req.diseases
    }
    to_query = {
        disease: list(dict.fromkeys(d.strip() for d in drugs))[:10]
        for disease, drugs in novel_by_disease.items()
    }
    if any(to_query.values()) and (not CATCHAT_BASE_URL or not CATCHAT_MODEL):
        first = next(d for drugs in to_query.values() for d in drugs)
        raise HTTPException(
            502,
            f"Novel drug scoring failed for '{first}': CatChat not configured "
            f"(CATCHAT_BASE_URL={CATCHAT_BASE_URL!r}, CATCHAT_MODEL={CATCHAT_MODEL!r})"
        )

    for drugs in to_query.values():
        for drug in drugs:
            progress.drugs.setdefault(drug, [0, 0])[1] += 2  # noCoT + CoT
    progress.total = 1 + sum(t for _, t in progress.drugs.values()) + len(req.diseases)
    progress.done = 1  # table lookup above
    progress.stage = "estimating"
    skipped_by_disease = await score_novel_drugs(to_query, budget, progress)
    progress.stage = "modeling"

    for disease in req.diseases:
        prevalence = PREVALENCES[disease]
        bundle = bundles[disease]
        feature_names = bundle.get("features", [])

        # 1. Encode demographics (validated in validate())
        age_val = int(req.demographics.get("age"))
        gender_val = GENDER_MAP.get(req.demographics.get("gender", "Man"), 0)
        race_val = RACE_MAP.get(req.demographics.get("race", "White"), 0)
        eth_val = ETHNICITY_MAP.get(req.demographics.get("ethnicity", "Others"), 0)

        # 2. Novel drugs were scored with CatChat above; drugs with no parseable
        #    probability are skipped (they contribute 0 to the DualR score, matching
        #    dualr_post.py dropna behavior).
        known = drug_probs.get(disease, {})
        novel_drugs = novel_by_disease[disease]
        skipped_drugs: list[str] = skipped_by_disease[disease]

        # 3. Compute DualR scores (all drugs now in table after fallback above)
        dualr_nocot = compute_dualr_score(req.drugs, disease, "nocot", prevalence)
        dualr_cot = compute_dualr_score(req.drugs, disease, "cot", prevalence)

        # 4. Build feature dict; bundle's feature_names determines column order.
        #    Include both naming conventions for the DualR features in case ml.py
        #    used "dualr_no_cot" or "dualr_nocot" — the bundle will pick the right one.
        feature_dict: dict = {
            "age": age_val,
            "gender": gender_val,
            "race": race_val,
            "ethnicity": eth_val,
            "dualr_no_cot": dualr_nocot,
            "dualr_nocot": dualr_nocot,
            "dualr_cot": dualr_cot,
        }
        for c in CHARLSON_COMORBIDITIES:
            feature_dict[c] = int(req.comorbidities.get(c, 0))

        # 5. Predict using the pipeline in the bundle
        row = pd.DataFrame([{k: feature_dict.get(k, 0) for k in feature_names}])
        risk = predict_risk(bundle, row)

        # 5a. Component contributions (marginal effects vs. all-zero baseline)
        dualr_cols = {"dualr_no_cot", "dualr_nocot", "dualr_cot"}
        demo_cols   = {"age", "gender", "race", "ethnicity"}

        neutral_row   = pd.DataFrame([{k: 0 for k in feature_names}])
        baseline_risk = predict_risk(bundle, neutral_row)

        drug_only_row = neutral_row.copy()
        for col in feature_names:
            if col in dualr_cols:
                drug_only_row[col] = row[col].values[0]
        drug_risk = predict_risk(bundle, drug_only_row)

        demo_only_row = neutral_row.copy()
        for col in feature_names:
            if col in demo_cols:
                demo_only_row[col] = row[col].values[0]
        demo_risk = predict_risk(bundle, demo_only_row)

        components = {
            "baseline":    round(baseline_risk, 4),
            "drug_effect": round(drug_risk - baseline_risk, 4),
            "demo_effect": round(demo_risk - baseline_risk, 4),
            "como_effect": round(risk - drug_risk - demo_risk + baseline_risk, 4),
        }

        # 6. Per-drug contributions for the results display
        top_drugs = []
        for drug in req.drugs:
            drug_clean = drug.strip()
            entry = drug_probs.get(disease, {}).get(drug_clean, {})
            contrib_nocot = 0.0
            contrib_cot = 0.0
            if "nocot" in entry:
                p = max(1e-10, min(1 - 1e-10, entry["nocot"]))
                contrib_nocot = float(np.clip(
                    np.log2(max(1e-10, (p / (1 - p)) / (prevalence / (1 - prevalence)))),
                    -LOG_OR_CLIP, LOG_OR_CLIP,
                ))
            if "cot" in entry:
                p = max(1e-10, min(1 - 1e-10, entry["cot"]))
                contrib_cot = float(np.clip(
                    np.log2(max(1e-10, (p / (1 - p)) / (prevalence / (1 - prevalence)))),
                    -LOG_OR_CLIP, LOG_OR_CLIP,
                ))
            top_drugs.append({
                "name": drug,
                "short_name": " ".join(drug.split()[:2]),
                "contribution_nocot": round(contrib_nocot, 3),
                "contribution_cot": round(contrib_cot, 3),
                "contribution_combined": round((contrib_nocot + contrib_cot) / 2, 3),
                "is_novel": drug_clean not in known,
                "is_skipped": drug_clean in skipped_drugs,
            })

        # Top 8 scored drugs by |contribution|, then all skipped drugs appended after
        scored = [d for d in top_drugs if not d["is_skipped"]]
        skipped_list = [d for d in top_drugs if d["is_skipped"]]
        scored.sort(key=lambda d: abs(d["contribution_combined"]), reverse=True)
        display_drugs = scored[:8] + skipped_list

        results[disease] = {
            "risk": round(risk, 4),
            "dualr_nocot": round(dualr_nocot, 3),
            "dualr_cot": round(dualr_cot, 3),
            "top_drugs": display_drugs,
            "components": components,
            "n_novel_drugs": len(novel_drugs),
            "n_known_drugs": len(req.drugs) - len(novel_drugs),
            "n_skipped_drugs": len(skipped_drugs),
            "skipped_drugs": skipped_drugs,
        }
        progress.tick()

    progress.stage = "done"
    return {"results": results}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Synchronous prediction, capped at CATCHAT_SYNC_BUDGET to stay under the proxy timeout."""
    validate(req)
    return await run_prediction(req, CATCHAT_SYNC_BUDGET, Progress())


# ═══════════════════════════════════════════
# Background jobs: start a prediction, poll its progress, fetch the result
# ═══════════════════════════════════════════

@dataclass
class Job:
    progress: Progress
    started: float
    task: asyncio.Task | None = None
    status: str = "running"  # running | done | error
    finished: float | None = None
    result: dict | None = None
    error: str | None = None


jobs: dict[str, Job] = {}


def _purge_jobs() -> None:
    """Drop finished jobs after JOB_TTL, and cancel any job that has run far too long."""
    now = time.monotonic()
    for job_id, job in list(jobs.items()):
        expired = job.finished is not None and now - job.finished > JOB_TTL
        stuck = job.finished is None and now - job.started > CATCHAT_JOB_BUDGET + JOB_TTL
        if expired or stuck:
            if job.task and not job.task.done():
                job.task.cancel()
            del jobs[job_id]


@app.post("/api/predict/jobs", status_code=202)
async def create_job(req: PredictRequest):
    _purge_jobs()
    validate(req)
    job_id = uuid.uuid4().hex
    job = Job(progress=Progress(), started=time.monotonic())
    jobs[job_id] = job

    async def runner():
        try:
            job.result = await run_prediction(req, CATCHAT_JOB_BUDGET, job.progress)
            job.status = "done"
        except asyncio.CancelledError:
            raise
        except HTTPException as e:
            job.status, job.error = "error", str(e.detail)
        except Exception:
            logger.exception(f"Prediction job {job_id} failed")
            job.status, job.error = "error", "Prediction failed. Please try again."
        finally:
            job.finished = time.monotonic()

    job.task = asyncio.create_task(runner())
    return {"job_id": job_id, "status": job.status, "progress": job.progress.as_dict()}


@app.get("/api/predict/jobs/{job_id}")
async def get_job(job_id: str):
    _purge_jobs()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This computation is no longer available. Please run it again.")
    body = {"job_id": job_id, "status": job.status, "progress": job.progress.as_dict()}
    if job.status == "done":
        body["result"] = job.result
    elif job.status == "error":
        body["error"] = job.error
    return body


@app.delete("/api/predict/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    job = jobs.pop(job_id, None)
    if job and job.task and not job.task.done():
        job.task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
