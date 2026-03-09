"""
train_model.py
==============
Training pipeline for the Phishing Detection System.

Trains four AI/ML classifiers, each grounded in a different syllabus concept,
then combines them into a soft-voting ensemble.

  Random Forest       (Unit 1 - Search): ensemble of decision trees built
                       via random feature subspacing — analogous to a
                       stochastic depth-limited DFS search space exploration.

  Naive Bayes         (Unit 4 - Probabilistic Reasoning): GaussianNB
                       implements the Bayesian Network classifier:
                       P(class | features) ∝ P(class) ∏ P(f_i | class).

  Gradient Boosting   (Unit 1 - Heuristic Search / Unit 4 - Probabilistic):
                       Greedy best-first gradient descent in function space;
                       each boosting stage corrects the residuals of the
                       previous stage (analogous to greedy hill-climbing).

  Decision Tree       (Unit 3 - CSP / Unit 2 - Logical Agents):
                       Produces an interpretable rule tree equivalent to a
                       set of propositional Horn clauses.  Each leaf is a
                       logical consequence of the path of conditions leading
                       to it (forward-chaining style).

  Voting Ensemble     (Unit 5 - Expert Systems): Combines the four classifiers
                       via soft-vote (average posterior probabilities), acting
                       as a meta-level expert that weighs evidence from all
                       reasoning modalities.
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from features import FEATURE_NAMES, extract_url_features


def build_feature_frame(urls: pd.Series) -> pd.DataFrame:
    """Apply extract_url_features to every URL; return aligned DataFrame."""
    rows = urls.apply(extract_url_features)
    return pd.DataFrame(rows.tolist())[FEATURE_NAMES]


def evaluate(name: str, model, X_test, y_test) -> dict:
    """Evaluate a model and print a report; return metrics dict."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 6),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 6),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 6),
    }
    print(f"\n── {name} ──")
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, digits=4))
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train phishing detection models incorporating AI syllabus concepts."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to dataset CSV with 'url' and 'status' columns",
    )
    parser.add_argument(
        "--out-dir", default="model",
        help="Output directory for saved model artefacts",
    )
    args = parser.parse_args()

    # ── Load & prepare dataset ───────────────────────────────────────────────
    print(f"\nLoading dataset: {args.data}")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["url", "status"]).reset_index(drop=True)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

    X = build_feature_frame(df["url"])
    y = (df["status"].str.lower() == "phishing").astype(int)
    print(f"  Features: {len(FEATURE_NAMES)}  |  Phishing: {y.sum()}  |  Legit: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Define classifiers ───────────────────────────────────────────────────
    # 1. Random Forest — Unit 1 (Search / Decision Trees)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # 2. Naive Bayes — Unit 4 (Probabilistic Reasoning / Bayesian Networks)
    nb = GaussianNB()

    # 3. Gradient Boosting — Unit 1 (Greedy Best-First heuristic search)
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )

    # 4. Decision Tree — Unit 2/3 (Logical Agent rules / CSP)
    dt = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    # 5. Soft-Voting Ensemble — Unit 5 (Expert System meta-reasoner)
    ensemble = VotingClassifier(
        estimators=[
            ("random_forest",       rf),
            ("naive_bayes",         nb),
            ("gradient_boosting",   gb),
            ("decision_tree",       dt),
        ],
        voting="soft",
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Train and evaluate each model individually ───────────────────────────
    all_metrics: dict = {}

    models_config = [
        ("Random Forest (Unit 1 - Search)",              rf,       "phishing_model_rf.joblib"),
        ("Naive Bayes (Unit 4 - Bayesian Network)",      nb,       "phishing_model_nb.joblib"),
        ("Gradient Boosting (Unit 1 - Greedy Search)",   gb,       "phishing_model_gb.joblib"),
        ("Decision Tree (Unit 2/3 - Logical/CSP)",       dt,       "phishing_model_dt.joblib"),
    ]

    print("\n═══ Training Individual Models ═══")
    for label, model, fname in models_config:
        print(f"\n▶ Training: {label}")
        model.fit(X_train, y_train)
        mname = fname.replace("phishing_model_", "").replace(".joblib", "")
        metrics = evaluate(label, model, X_test, y_test)
        all_metrics[mname] = metrics
        path = os.path.join(args.out_dir, fname)
        joblib.dump(model, path)
        print(f"  Saved → {path}")

    # ── Train and evaluate the voting ensemble ───────────────────────────────
    print("\n═══ Training Voting Ensemble (Unit 5 - Expert System) ═══")
    ensemble.fit(X_train, y_train)
    ensemble_metrics = evaluate(
        "Soft-Voting Ensemble", ensemble, X_test, y_test
    )
    all_metrics["ensemble"] = ensemble_metrics
    ensemble_path = os.path.join(args.out_dir, "phishing_model_ensemble.joblib")
    joblib.dump(ensemble, ensemble_path)
    print(f"\n  Saved → {ensemble_path}")

    # ── Feature importance (from Random Forest) ──────────────────────────────
    importances = dict(zip(
        FEATURE_NAMES,
        [round(float(v), 6) for v in rf.feature_importances_],
    ))
    importances_sorted = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        "feature_names":      FEATURE_NAMES,
        "feature_count":      len(FEATURE_NAMES),
        "models_trained": [
            "phishing_model_rf.joblib       (Random Forest — Unit 1 Search)",
            "phishing_model_nb.joblib       (Naive Bayes   — Unit 4 Probabilistic)",
            "phishing_model_gb.joblib       (Gradient Boost— Unit 1 Greedy Search)",
            "phishing_model_dt.joblib       (Decision Tree — Unit 2/3 Logical/CSP)",
            "phishing_model_ensemble.joblib (Soft-Vote Ensemble — Unit 5 Expert Sys)",
        ],
        "primary_model":      "phishing_model_ensemble.joblib",
        "metrics":            all_metrics,
        "feature_importances": importances_sorted,
        "ai_concepts": {
            "Unit1_Intelligent_Agents": [
                "PEAS framework (PhishingDetectionAgent in ai_agent.py)",
                "Random Forest as stochastic DFS tree search ensemble",
                "Gradient Boosting as greedy best-first descent in function space",
            ],
            "Unit1_Search_Strategies": [
                "BFS / DFS / UCS over URLComponentGraph",
                "Greedy Best-First search with suspicion heuristic",
                "A* search with g(n)+h(n) on URL graph nodes",
            ],
            "Unit2_Logical_Agents": [
                "Forward Chaining rule engine (PHISHING_RULES knowledge base)",
                "Backward Chaining explainer (explanation facility)",
                "Horn clause / FOL representation of URL predicates",
                "Decision Tree encodes propositional IF-THEN rules",
            ],
            "Unit3_CSP_Knowledge": [
                "UrlConstraintSatisfaction: Variables, Domains, Constraints",
                "15 URL legitimacy constraints as arc-consistency checks",
                "PhishingKnowledgeBase: working memory + long-term rule store",
            ],
            "Unit4_Probabilistic": [
                "GaussianNB as Bayesian Network classifier",
                "BayesianURLReasoner: CPTs + log-space Bayes inference",
                "Shannon entropy feature (information theory)",
                "Bigram anomaly: HMM-inspired character transition scoring",
            ],
            "Unit5_NLP_ExpertSystems": [
                "NLP features: entropy, vowel ratio, avg token length, bigrams",
                "Brand impersonation detection (NLP pattern matching)",
                "Suspicious TLD knowledge base",
                "VotingClassifier as Expert System meta-reasoner",
                "BackwardChainExplainer as Expert System explanation facility",
            ],
        },
    }
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved metadata → {meta_path}")

    print("\n══════════════════════════════════════")
    print("  Training complete.")
    print(f"  Ensemble F1   : {ensemble_metrics['f1']:.4f}")
    print(f"  Ensemble Acc  : {ensemble_metrics['accuracy']:.4f}")
    print("══════════════════════════════════════\n")


if __name__ == "__main__":
    main()

