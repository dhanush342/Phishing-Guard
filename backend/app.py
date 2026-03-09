"""
app.py
======
FastAPI REST API for the Phishing Detection System.

Loads the soft-voting ensemble model (trained from train_model.py) and
exposes endpoints that combine:
  - ML model prediction (ensemble of RF, NB, GB, DT)
  - AI agent reasoning (BFS/DFS/A* graph analysis)
  - Forward/Backward chaining rule engine (Expert System)
  - CSP constraint violation report
  - Bayesian probability estimate
  - NLP URL feature analysis
"""

import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import FEATURE_NAMES, extract_url_features
from ai_agent import (
    PhishingDetectionAgent,
    ForwardChainRuleEngine,
    BackwardChainExplainer,
    UrlConstraintSatisfaction,
    BayesianURLReasoner,
)

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_DIR      = os.path.join(os.path.dirname(__file__), "model")
ENSEMBLE_PATH  = os.path.join(MODEL_DIR, "phishing_model_ensemble.joblib")
RF_PATH        = os.path.join(MODEL_DIR, "phishing_model_rf.joblib")
META_PATH      = os.path.join(MODEL_DIR, "metadata.json")

def _load_model():
    """Load ensemble; fall back to random forest if ensemble not available."""
    if os.path.exists(ENSEMBLE_PATH):
        return joblib.load(ENSEMBLE_PATH), "ensemble"
    if os.path.exists(RF_PATH):
        return joblib.load(RF_PATH), "random_forest"
    raise RuntimeError(
        "No trained model found. Run train_model.py before starting the API."
    )

model, model_type = _load_model()

metadata: dict = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Phishing Detection System",
    description=(
        "Intelligent phishing URL detection powered by:\n"
        "• Soft-Voting Ensemble (RF + NaiveBayes + GradientBoost + DecisionTree)\n"
        "• A* / BFS / DFS search on URL component graph\n"
        "• Forward/Backward chaining rule engine\n"
        "• CSP constraint checking\n"
        "• Bayesian Network probabilistic reasoning\n"
        "• NLP + HMM-inspired URL features"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise AI agent modules (stateless — safe to share)
_agent      = PhishingDetectionAgent()
_csp        = UrlConstraintSatisfaction()
_rule_engine= ForwardChainRuleEngine()
_explainer  = BackwardChainExplainer()
_bayesian   = BayesianURLReasoner()


# ── Request / Response models ─────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    url: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":          "Phishing Detection System API",
        "version":       "2.0.0",
        "status":        "running",
        "model_loaded":  model_type,
        "feature_count": len(FEATURE_NAMES),
        "docs":          "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_type}


@app.get("/model-info")
def model_info():
    """Return model metadata including metrics and AI concepts used."""
    return metadata if metadata else {"detail": "metadata not available"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Analyse a URL and return:
      - ML ensemble prediction + probability
      - AI agent reasoning trace (rules fired, CSP violations, Bayesian score)
      - URL graph search summary (BFS/DFS/A*)
      - Natural-language backward-chain explanation
    """
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    # 1. Extract features (sensor / percept)
    features       = extract_url_features(url)
    feature_frame  = pd.DataFrame([features])[FEATURE_NAMES]

    # 2. ML ensemble prediction
    probability = 0.0
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(feature_frame)[0][1])
    else:
        probability = float(model.predict(feature_frame)[0])

    label = "phishing" if probability >= 0.5 else "legitimate"

    # 3. Forward chaining rule engine (Expert System inference)
    rules_fired, rule_score = _rule_engine.run(features)

    # 4. CSP constraint checking
    csp_violations = _csp.check_violations(features)
    csp_score      = _csp.satisfaction_score(features)

    # 5. Bayesian probabilistic reasoning
    bayesian_score = _bayesian.score(features)

    # 6. Backward chaining explanation
    explanations = _explainer.explain(features)

    # 7. URL graph search summary (BFS / DFS / A*)
    from ai_agent import URLComponentGraph
    graph              = URLComponentGraph(url)
    bfs_nodes          = graph.bfs()
    dfs_nodes          = graph.dfs()
    a_star_nodes       = graph.a_star()
    greedy_nodes       = graph.greedy_best_first()
    suspicious_nodes   = graph.suspicious_node_count()

    return {
        # ── ML prediction ──────────────────────────────────────────────────
        "label":             label,
        "ml_probability":    round(probability, 4),
        "model_used":        model_type,

        # ── Key URL features ───────────────────────────────────────────────
        "url_features": {
            "url_length":         features["url_length"],
            "has_ip":             bool(features["has_ip"]),
            "has_https":          bool(features["has_https"]),
            "at_sign":            features["at_count"] > 0,
            "http_in_path":       bool(features["has_http_in_path"]),
            "shortening_service": bool(features["shortening_service"]),
            "suspicious_words":   features["suspicious_words"],
            "subdomain_count":    features["subdomain_count"],
            "entropy":            features["entropy"],
            "bigram_anomaly":     features["bigram_anomaly"],
            "brand_impersonation":bool(features["brand_impersonation"]),
            "suspicious_tld":     bool(features["has_suspicious_tld"]),
        },

        # ── Expert System / Forward Chaining ──────────────────────────────
        "rule_engine": {
            "rules_fired_count": len(rules_fired),
            "rule_score":        rule_score,
            "rules_fired":       rules_fired,
        },

        # ── CSP Analysis ──────────────────────────────────────────────────
        "csp_analysis": {
            "violations_count":  len(csp_violations),
            "satisfaction_score": round(csp_score, 4),
            "violations":        csp_violations,
        },

        # ── Bayesian Network ──────────────────────────────────────────────
        "bayesian_score": bayesian_score,

        # ── URL Graph Search ──────────────────────────────────────────────
        "graph_search": {
            "total_nodes":       len(graph.nodes),
            "suspicious_nodes":  suspicious_nodes,
            "bfs_order":         bfs_nodes[:10],
            "dfs_order":         dfs_nodes[:10],
            "a_star_top5":       [(round(f, 2), n) for f, n in a_star_nodes[:5]],
            "greedy_top5":       greedy_nodes[:5],
        },

        # ── Backward Chaining Explanation ─────────────────────────────────
        "explanations":          explanations,
    }


@app.get("/analyze")
def analyze_get(url: str):
    """GET-friendly alias for /predict (URL passed as query param)."""
    return predict(PredictionRequest(url=url))

