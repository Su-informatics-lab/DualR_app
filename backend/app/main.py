"""
DualR Backend — FastAPI
Prediction endpoint using exported XGBoost models and DualR lookup tables.

Feature schema (must match ml.py exactly):
- age: numeric integer [18, 120] — passed directly as a continuous feature
- race, ethnicity, gender: categorical-encoded integers (see maps below)
- Charlson comorbidities: disease-specific binary subsets
- dualr_nocot, dualr_cot: continuous scores computed from drug lookup tables
"""

import os
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import xgboost as xgb
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# Feature Definitions (from ml.py)
# ═══════════════════════════════════════════

DEMO_FEATURES = ["age", "race", "ethnicity", "gender"]

CHARLSON_COMORBIDITIES = [
    "HIV", "AIDS", "Cerebrovascular_Disease", "Congestive_Heart_Failure",
    "Myocardial_Infarction", "Peripheral_Vascular_Disease",
    "Chronic_Pulmonary_Disease", "Dementia", "Liver_Disease_Mild",
    "Liver_Disease_Moderate_Severe", "Malignancy", "Metastatic_Solid_Tumor",
    "Peptic_Ulcer_Disease", "Renal_Disease_Mild_Moderate",
    "Renal_Disease_Severe", "Rheumatic_Disease", "Hemiplegia_Paraplegia",
    "Diabetes_with_Chronic_Complications", "Diabetes_without_Chronic_Complications",
]

DISEASE_FEATURE_MAP = {
    "t2d": (
        DEMO_FEATURES,
        [c for c in CHARLSON_COMORBIDITIES if not c.startswith("Diabetes")],
    ),
    "htn": (DEMO_FEATURES, CHARLSON_COMORBIDITIES),
    "aud": (DEMO_FEATURES, CHARLSON_COMORBIDITIES),
}

PREVALENCES = {"t2d": 0.109, "htn": 0.330, "aud": 0.078}

# Categorical encoding (matches ml.py reference encoding)
GENDER_MAP = {"Man": 0, "Woman": 1, "Other": 2}
RACE_MAP = {"White": 0, "Black": 1, "Others": 2}
ETHNICITY_MAP = {"Others": 0, "Hispanic": 1}

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# ═══════════════════════════════════════════
# Global State (loaded at startup)
# ═══════════════════════════════════════════

models: dict = {}       # disease -> xgb.Booster
drug_probs: dict = {}   # disease -> {drug_name -> {"nocot": p, "cot": p}}


def load_models():
    """Load XGBoost models and drug probability tables at startup."""
    for disease in ["t2d", "htn", "aud"]:
        model_path = os.path.join(MODEL_DIR, f"xgb_{disease}.json")
        if os.path.exists(model_path):
            bst = xgb.Booster()
            bst.load_model(model_path)
            models[disease] = bst
            logger.info(f"Loaded XGBoost model: {model_path}")
        else:
            logger.warning(f"Model not found: {model_path}")

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
    Compute patient-level DualR score from drug list.
    Matches dualr_post.py: sum of log2 odds ratios relative to prevalence.
    """
    log_odds = []
    probs_table = drug_probs.get(disease, {})

    for drug in drug_names:
        drug_clean = drug.strip()
        if drug_clean in probs_table and mode in probs_table[drug_clean]:
            p = probs_table[drug_clean][mode]
            p = max(1e-10, min(1 - 1e-10, p))
            drug_odds = p / (1 - p)
            base_odds = baseline_prob / (1 - baseline_prob)
            or_val = drug_odds / base_odds
            or_val = max(1e-10, or_val)
            log_odds.append(np.log2(or_val))

    if not log_odds:
        return 0.0
    return sum(log_odds)


async def query_novel_drug_prob(drug_name: str, disease: str, use_cot: bool) -> float:
    """Query vLLM for P(disease|drug) for an unseen drug."""
    disease_names = {"t2d": "type 2 diabetes", "htn": "hypertension", "aud": "alcohol use disorder"}
    disease_full = disease_names.get(disease, disease)

    if use_cot:
        prompt = (
            f"Given that a patient took {drug_name}, estimate the probability that "
            f"they have {disease_full}. Think step by step, then provide your final "
            f"answer as a decimal between 0 and 1."
        )
    else:
        prompt = (
            f"Given that a patient took {drug_name}, estimate the probability that "
            f"they have {disease_full}. Respond with only a decimal number between 0 and 1."
        )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512 if use_cot else 16,
                    "temperature": 0.01,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            import re
            numbers = re.findall(r"0\.\d+", text)
            if numbers:
                return float(numbers[-1])
            return PREVALENCES.get(disease, 0.1)
    except Exception as e:
        logger.error(f"vLLM query failed for {drug_name}: {e}")
        return PREVALENCES.get(disease, 0.1)


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
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    results = {}

    for disease in req.diseases:
        if disease not in DISEASE_FEATURE_MAP:
            raise HTTPException(400, f"Unknown disease: {disease}")

        demo_feats, como_feats = DISEASE_FEATURE_MAP[disease]
        prevalence = PREVALENCES[disease]

        # 1. Encode demographics
        try:
            age_val = int(req.demographics.get("age"))
        except (TypeError, ValueError):
            raise HTTPException(400, "demographics.age must be an integer")
        if not (18 <= age_val <= 120):
            raise HTTPException(400, f"demographics.age must be 18–120, got {age_val}")
        gender_val = GENDER_MAP.get(req.demographics.get("gender", "Man"), 0)
        race_val = RACE_MAP.get(req.demographics.get("race", "White"), 0)
        eth_val = ETHNICITY_MAP.get(req.demographics.get("ethnicity", "Others"), 0)

        demo_vector = []
        for f in demo_feats:
            if f == "age":
                demo_vector.append(age_val)
            elif f == "gender":
                demo_vector.append(gender_val)
            elif f == "race":
                demo_vector.append(race_val)
            elif f == "ethnicity":
                demo_vector.append(eth_val)

        # 2. Encode comorbidities (binary, disease-specific subset)
        como_vector = [int(req.comorbidities.get(c, 0)) for c in como_feats]

        # 3. Compute DualR scores
        dualr_nocot = compute_dualr_score(req.drugs, disease, "nocot", prevalence)
        dualr_cot = compute_dualr_score(req.drugs, disease, "cot", prevalence)

        # 4. Check for novel drugs and query vLLM if needed
        known_drugs = drug_probs.get(disease, {})
        novel_drugs = [d for d in req.drugs if d.strip() not in known_drugs]
        if novel_drugs:
            logger.info(f"Querying {len(novel_drugs)} novel drugs via vLLM for {disease}")
            for drug in novel_drugs[:10]:  # Limit to 10 novel drugs
                p_nocot = await query_novel_drug_prob(drug, disease, use_cot=False)
                p_cot = await query_novel_drug_prob(drug, disease, use_cot=True)
                for p, score_ref in [(p_nocot, "nocot"), (p_cot, "cot")]:
                    p = max(1e-10, min(1 - 1e-10, p))
                    drug_odds = p / (1 - p)
                    base_odds = prevalence / (1 - prevalence)
                    lo = np.log2(max(1e-10, drug_odds / base_odds))
                    if score_ref == "nocot":
                        dualr_nocot += lo
                    else:
                        dualr_cot += lo

        # 5. Assemble feature vector: demo + como + dualr_nocot + dualr_cot
        feature_vector = demo_vector + como_vector + [dualr_nocot, dualr_cot]

        # 6. Predict with XGBoost
        if disease in models:
            dmatrix = xgb.DMatrix(np.array([feature_vector]))
            risk = float(models[disease].predict(dmatrix)[0])
        else:
            combined = (dualr_nocot + dualr_cot) / 2
            risk = 1 / (1 + np.exp(-combined * 0.3))
            logger.warning(f"Using fallback prediction for {disease} (model not loaded)")

        # 7. Per-drug contributions (for interpretability display)
        top_drugs = []
        for drug in req.drugs:
            drug_clean = drug.strip()
            contrib_nocot = 0.0
            contrib_cot = 0.0
            if drug_clean in known_drugs:
                probs = known_drugs[drug_clean]
                if "nocot" in probs:
                    p = max(1e-10, min(1 - 1e-10, probs["nocot"]))
                    contrib_nocot = np.log2(max(1e-10, (p / (1 - p)) / (prevalence / (1 - prevalence))))
                if "cot" in probs:
                    p = max(1e-10, min(1 - 1e-10, probs["cot"]))
                    contrib_cot = np.log2(max(1e-10, (p / (1 - p)) / (prevalence / (1 - prevalence))))

            top_drugs.append({
                "name": drug,
                "short_name": " ".join(drug.split()[:2]),
                "contribution_nocot": round(contrib_nocot, 3),
                "contribution_cot": round(contrib_cot, 3),
                "contribution_combined": round((contrib_nocot + contrib_cot) / 2, 3),
                "is_novel": drug_clean not in known_drugs,
            })

        top_drugs.sort(key=lambda d: abs(d["contribution_combined"]), reverse=True)

        results[disease] = {
            "risk": round(float(risk), 4),
            "dualr_nocot": round(dualr_nocot, 3),
            "dualr_cot": round(dualr_cot, 3),
            "top_drugs": top_drugs[:8],
            "n_novel_drugs": len(novel_drugs),
            "n_known_drugs": len(req.drugs) - len(novel_drugs),
        }

    return PredictResponse(results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
