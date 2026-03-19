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

import json
import logging
import os
import re
from contextlib import asynccontextmanager
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

# Categorical encoding (matches ml.py reference encoding)
GENDER_MAP = {"Man": 0, "Woman": 1, "Other": 2}
RACE_MAP = {"White": 0, "Black": 1, "Others": 2}
ETHNICITY_MAP = {"Others": 0, "Hispanic": 1}

MODEL_DIR = os.getenv("MODEL_DIR", "models")
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/dualr_cache")

# CatChat (MSU) — backend-only fallback for novel drugs
CATCHAT_BASE_URL = os.getenv("CATCHAT_BASE_URL", "")
CATCHAT_MODEL = os.getenv("CATCHAT_MODEL", "")
CATCHAT_API_KEY = os.getenv("CATCHAT_API_KEY", "")

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
    Sum of log2 odds ratios for known drugs relative to disease prevalence.
    Matches dualr_post.py aggregation.
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
            log_odds.append(np.log2(or_val))
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


async def query_catchat(drug_name: str, disease: str, use_cot: bool) -> float | None:
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{CATCHAT_BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": CATCHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512 if use_cot else 16,
                    "temperature": 0.01,
                    **( {"reasoning_effort": "medium"} if "oss" in CATCHAT_MODEL.lower() else {} ),
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            numbers = re.findall(r"0\.\d+", text)
            if numbers:
                return float(numbers[-1])
            logger.warning(
                f"CatChat returned no parseable probability for drug={drug_name}, "
                f"disease={disease}; skipping. Raw: {text[:200]}"
            )
            return None
    except Exception as e:
        logger.error(f"CatChat query failed for drug={drug_name}, disease={disease}: {e}")
        return None


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


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    results = {}

    for disease in req.diseases:
        if disease not in PREVALENCES:
            raise HTTPException(400, f"Unknown disease: {disease}")
        if disease not in bundles:
            raise HTTPException(503, f"Model not loaded for disease: {disease}")

        prevalence = PREVALENCES[disease]
        bundle = bundles[disease]
        feature_names = bundle.get("features", [])

        # 1. Validate and encode demographics
        try:
            age_val = int(req.demographics.get("age"))
        except (TypeError, ValueError):
            raise HTTPException(400, "demographics.age must be an integer")
        if not (18 <= age_val <= 120):
            raise HTTPException(400, f"demographics.age must be 18–120, got {age_val}")
        gender_val = GENDER_MAP.get(req.demographics.get("gender", "Man"), 0)
        race_val = RACE_MAP.get(req.demographics.get("race", "White"), 0)
        eth_val = ETHNICITY_MAP.get(req.demographics.get("ethnicity", "Others"), 0)

        # 2. Novel drugs: query CatChat and add to drug_probs before scoring.
        #    Drugs where CatChat returns no parseable probability are skipped
        #    (they contribute 0 to the DualR score, matching dualr_post.py dropna behavior).
        known = drug_probs.get(disease, {})
        novel_drugs = [d for d in req.drugs if d.strip() not in known]
        skipped_drugs: list[str] = []
        if novel_drugs:
            logger.info(f"Novel drugs for {disease}: {len(novel_drugs)}")
            for drug in novel_drugs[:10]:  # cap at 10 per request
                drug_clean = drug.strip()
                scored = False
                try:
                    for use_cot, mode in [(False, "nocot"), (True, "cot")]:
                        p = await query_catchat(drug_clean, disease, use_cot)
                        if p is not None:
                            if drug_clean not in drug_probs[disease]:
                                drug_probs[disease][drug_clean] = {}
                            drug_probs[disease][drug_clean][mode] = p
                            _write_cache(drug_clean, disease, mode, p)
                            scored = True
                except RuntimeError as e:
                    raise HTTPException(
                        502,
                        f"Novel drug scoring failed for '{drug_clean}': {e}"
                    )
                if not scored:
                    skipped_drugs.append(drug_clean)
                    logger.info(f"Skipped novel drug (no probability): {drug_clean}")

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
        risk = float(bundle["pipeline"].predict_proba(row)[0][1])

        # 6. Per-drug contributions for the results display
        top_drugs = []
        for drug in req.drugs:
            drug_clean = drug.strip()
            entry = drug_probs.get(disease, {}).get(drug_clean, {})
            contrib_nocot = 0.0
            contrib_cot = 0.0
            if "nocot" in entry:
                p = max(1e-10, min(1 - 1e-10, entry["nocot"]))
                contrib_nocot = np.log2(max(1e-10, (p / (1 - p)) / (prevalence / (1 - prevalence))))
            if "cot" in entry:
                p = max(1e-10, min(1 - 1e-10, entry["cot"]))
                contrib_cot = np.log2(max(1e-10, (p / (1 - p)) / (prevalence / (1 - prevalence))))
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
            "n_novel_drugs": len(novel_drugs),
            "n_known_drugs": len(req.drugs) - len(novel_drugs),
            "n_skipped_drugs": len(skipped_drugs),
            "skipped_drugs": skipped_drugs,
        }

    return PredictResponse(results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
