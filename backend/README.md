# Phishing Detection System — Backend

An intelligent phishing URL detection system that integrates **all five AI syllabus units**
into a single, production-ready API.  
Every classifier, reasoning module, and feature is grounded in a formal AI concept.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (from the backend/ folder)
python train_model.py --data "..\dataset_phishing (1).csv" --out-dir model

# 3. Start the API
uvicorn app:app --host 127.0.0.1 --port 8000

# 4. Open index.html in a browser and analyse URLs
```

---

## Architecture Overview

```
URL (raw string)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  features.py  —  Feature Extraction Layer (29 features)            │
│  • Structural / lexical features (23)                               │
│  • NLP / entropy / HMM-inspired features (6)                       │
└─────────────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────────────┐
       │                                                              │
       ▼                                                              ▼
┌────────────────────────────┐          ┌──────────────────────────────┐
│  ai_agent.py — Reasoning   │          │  ML Ensemble Model           │
│  • BFS / DFS / UCS         │          │  RF + NB + GB + DT           │
│  • A* / Greedy Best-First  │          │  Soft-Voting Classifier      │
│  • CSP Constraint Check    │          └──────────────────────────────┘
│  • Forward Chaining Rules  │
│  • Backward Chain Explain  │
│  • Bayesian Network Score  │
│  • NLP URL Analyser        │
└────────────────────────────┘
       │                                        │
       └──────────────────────┬─────────────────┘
                              ▼
                    ┌─────────────────┐
                    │    app.py API   │
                    │  /predict POST  │
                    └─────────────────┘
```

---

## AI Concepts Used — Full Mapping to Syllabus

---

### Unit 1 — Intelligent Agents & Search Strategies

#### 1.1 Intelligent Agents (PEAS Framework)
**File:** `ai_agent.py` → `PhishingDetectionAgent`

The phishing detector is modelled as a **rational agent** using the PEAS framework:

| PEAS Component | Implementation |
|---|---|
| **P**erformance Measure | Accuracy, F1-score, minimal false-negative rate |
| **E**nvironment | Internet URL space (partially observable, stochastic, episodic) |
| **A**ctuators | Classification label + probability + natural-language explanation |
| **S**ensors | Raw URL string → `extract_url_features()` → 29-dimensional percept |

The agent is **model-based**: it maintains an internal model (the trained ML ensemble) combined with a rule-based knowledge base, making it more than a simple reflex agent.

#### 1.2 Uninformed Search Strategies
**File:** `ai_agent.py` → `URLComponentGraph`

Every URL is parsed into a **directed acyclic graph** where nodes represent scheme, host parts, path segments, and query parameters. Three uninformed strategies traverse this graph:

| Strategy | Class Method | Purpose |
|---|---|---|
| **Breadth-First Search (BFS)** | `graph.bfs()` | Explores URL components level by level; finds all nodes at each structural depth before going deeper |
| **Depth-First Search (DFS)** | `graph.dfs()` | Follows a single URL path as deep as possible; useful for detecting deeply nested suspicious path segments |
| **Uniform-Cost Search (UCS)** | `graph.ucs()` | Assigns cost = number of edges from ROOT; expands minimum-cost nodes first; guarantees optimal path ordering |

#### 1.3 Informed (Heuristic) Search Strategies
**File:** `ai_agent.py` → `URLComponentGraph`

Two heuristic-driven strategies prioritise suspicious URL components:

| Strategy | Class Method | Heuristic h(n) |
|---|---|---|
| **Greedy Best-First** | `graph.greedy_best_first()` | Pure h(n): counts suspicious keywords in node label; always expands the most suspicious node next — fast but not optimal |
| **A\* Search** | `graph.a_star()` | f(n) = g(n) + h(n); g(n) = path cost (depth), h(n) = keyword suspicion score; optimal and complete when h is admissible |

The heuristic `h(n)` is **admissible**: it never overestimates because a node can contain at most as many suspicious keywords as actually present.

#### 1.4 Depth-Limited & Iterative Deepening
The `DecisionTreeClassifier` (Unit 2/3 model) enforces a `max_depth=12` parameter, directly implementing the **depth-limited search** concept. At inference time, each prediction traverses a path of depth ≤ 12 — equivalent to a depth-limited DFS over the learned rule tree.

#### 1.5 Constraint Satisfaction Problem (CSP)
**File:** `ai_agent.py` → `UrlConstraintSatisfaction`

URL legitimacy is formulated as a CSP:

- **Variables**: URL structural attributes (length, has_ip, https, at_count, …)
- **Domains**: Acceptable value ranges for legitimate URLs
- **Constraints (15 total)**: Logical conditions that must hold simultaneously

```
C01: url_length ≤ 75             C09: not a shortening service
C02: hostname ≠ raw IP address   C10: suspicious_words ≤ 1
C03: scheme = HTTPS              C11: digit_ratio ≤ 35%
C04: no '@' in URL               C12: Shannon entropy ≤ 4.5
C05: no HTTP embedded in path    C13: bigram_anomaly ≤ 0.75
C06: no '//' after scheme        C14: TLD not in high-risk list
C07: subdomain_depth ≤ 2         C15: no brand impersonation
C08: no punycode hostname
```

Each violated constraint is an **arc-inconsistency** — direct evidence of a phishing URL. The CSP satisfaction score (`violations / total`) feeds into the final assessment.

---

### Unit 2 — Logical Agents & Reasoning

#### 2.1 Knowledge Base (Propositional / First-Order Logic)
**File:** `ai_agent.py` → `PHISHING_RULES`

19 Horn clauses form the knowledge base. Each rule maps a URL predicate to a phishing consequence, expressed in First-Order Logic notation:

```
R01: HasIPAddress(url)          → SuspectHost(url)             weight=3
R02: ¬UsesHTTPS(url)            → InsecureChannel(url)          weight=2
R03: ContainsAt(url)            → CredentialHiding(url)         weight=3
R04: HTTPInPath(url)            → OpenRedirect(url)             weight=2
R06: SuspicionTokens(url,N)∧N≥2 → SocialEngineering(url)        weight=3
R07: PunycodeHost(url)          → HomographAttack(url)          weight=3
R11: HasIPAddress∧¬HTTPS(url)   → HighRiskHost(url)             weight=2
R17: HighBigramAnomaly(url)     → AutoGeneratedDomain(url)      weight=2
R18: SuspiciousTLD(url)         → HighRiskRegistrar(url)        weight=2
R19: BrandInSubdomainOrPath(url)→ Impersonation(url)            weight=3
… (9 more rules)
```

#### 2.2 Forward Chaining
**File:** `ai_agent.py` → `ForwardChainRuleEngine`

Implements **data-driven (bottom-up) Modus Ponens**:
1. Load all URL feature values as **facts** into working memory.
2. Scan every rule: if `antecedent(facts) == True` → **fire** the rule.
3. Accumulate weighted evidence; normalise total to [0, 1].

This is equivalent to the TELL/ASK interface of a logical agent's knowledge base.

#### 2.3 Backward Chaining
**File:** `ai_agent.py` → `BackwardChainExplainer`

Implements **goal-driven (top-down) inference**:
1. Goal: `Phishing(url)` is asserted.
2. Find all rules whose consequent supports `Phishing`.
3. Verify each rule's antecedent against the known facts.
4. Return a natural-language justification for every confirmed inference chain.

This constitutes the **Explanation Facility** of the Expert System: answering *"Why is this URL phishing?"*

#### 2.4 Unification & Resolution
The `DecisionTreeClassifier` at inference time performs logical **resolution**: each internal node is a propositional test (e.g., `entropy ≤ 4.2`), and the path from root to leaf constitutes a **resolution proof** that the URL belongs to a class.

---

### Unit 3 — Knowledge Representation & Reasoning

#### 3.1 State-Space Planning
**File:** `ai_agent.py` → `PhishingKnowledgeBase`

The `PhishingKnowledgeBase` maintains:
- **Working memory** (facts = observed feature values for a URL)
- **Long-term memory** (rules = PHISHING_RULES knowledge base)
- **Inferred facts** (conclusions derived by running the forward-chaining engine)

This mirrors a planning agent that starts in an initial state (URL features), applies operators (rules), and produces a goal state (phishing conclusion).

#### 3.2 Partial-Order Planning (Feature Inference Pipeline)
The agent acts as a partial-order planner: feature extraction, rule evaluation, CSP checking, and Bayesian scoring are partially ordered steps (some can proceed in any order; CSP and Bayesian scoring are independent of each other).

#### 3.3 Knowledge Base Feature Files
`features.py` exports two structured knowledge bases:
- `SUSPICIOUS_TLDS` — high-risk TLD set (domain knowledge)
- `BRAND_NAMES` — impersonation detection list
- `LEGIT_CHAR_BIGRAMS` — HMM emission-model bigrams for legitimate URLs

---

### Unit 4 — Uncertainty & Probabilistic Reasoning

#### 4.1 Bayesian Network
**File:** `ai_agent.py` → `BayesianURLReasoner`

A **Naive Bayesian Network** (class-conditional independence) with:
- **Prior**: P(Phishing) = 0.50 (balanced dataset)
- **Conditional Probability Tables (CPTs)** for 11 feature nodes:

```
Node              P(feat=1|Phishing)   P(feat=1|Legitimate)
has_ip            0.38                 0.01
has_https         0.18 (low!)          0.80
brand_impersonation 0.28               0.02
has_suspicious_tld 0.30               0.04
…
```

Inference uses **log-space Bayes** to prevent numerical underflow:
```
log P(Phishing|F) = log P(Phishing) + Σ log P(fi | Phishing)
```
Posterior is normalised via **log-sum-exp** (Variable Elimination style).

#### 4.2 GaussianNB Classifier (Probabilistic Model)
**File:** `train_model.py` — `GaussianNB`

Fits `P(feature_i | class) = N(μ, σ²)` for each feature under each class.
At prediction: returns exact Bayesian posterior probability for the URL.

#### 4.3 Shannon Entropy (Information Theory)
**File:** `features.py` → `_shannon_entropy()`

```
H(URL) = -Σ p(c) · log₂ p(c)   over character distribution
```
High entropy URLs are characteristic of auto-generated phishing subdomains (e.g., `a3xk9p.malicious.xyz`). Threshold: H > 4.5 triggers rule R16.

#### 4.4 Hidden Markov Model — Bigram Character Transitions
**File:** `features.py` → `_bigram_anomaly()`, `LEGIT_CHAR_BIGRAMS`

The URL character sequence is modelled as an observed HMM emission sequence.
The `LEGIT_CHAR_BIGRAMS` set acts as the emission probability matrix for the "Legitimate" hidden state.

```
bigram_anomaly = |{bigrams not in LEGIT_BIGRAMS}| / |all_bigrams|
```
Auto-generated phishing domains (e.g., `xk3p9.attacker.com`) have very few matching bigrams → high anomaly score → triggers rule R17.

---

### Unit 5 — Applications: NLP & Expert Systems

#### 5.1 Natural Language Processing — URL as Text
**File:** `features.py` → NLP feature functions; `ai_agent.py` → `NLPUrlFeatureExtractor`

The URL is treated as a **text document** and processed with NLP techniques:

| NLP Feature | Method | Description |
|---|---|---|
| `entropy` | Shannon H(X) | Character-level entropy — measures lexical randomness |
| `vowel_ratio` | _vowel_ratio() | Linguistic normality — legitimate domain names follow human language vowel patterns |
| `avg_token_length` | _avg_token_len() | Tokenises URL on `[a-zA-Z0-9]+`; short avg lengths suggest random substrings |
| `bigram_anomaly` | _bigram_anomaly() | HMM-inspired bigram emission scoring (see Unit 4.4) |
| `brand_impersonation` | _brand_impersonation() | NLP pattern matching: brand name appears in wrong URL position |
| `suspicious_words` | keyword scan | Counts phishing-indicator tokens (login, verify, secure, paypal, …) |

#### 5.2 Parsing
The URL parser (`urllib.parse.urlparse`) performs **syntactic parsing** of the URL grammar:
```
URL → scheme "://" authority path ["?" query]
authority → [userinfo "@"] host [":" port]
host → subdomain* "." domain "." tld
```
This structural decomposition enables feature extraction and graph construction.

#### 5.3 Expert System Architecture
**File:** `ai_agent.py` — complete Expert System implementation

The system follows the classic Expert System architecture:

| Component | Implementation |
|---|---|
| **Knowledge Base** | `PHISHING_RULES` — 19 Horn-clause rules + 2 knowledge-base sets |
| **Inference Engine** | `ForwardChainRuleEngine` — forward-chaining Modus Ponens |
| **Working Memory** | `PhishingKnowledgeBase.facts` — current URL feature values |
| **Explanation Facility** | `BackwardChainExplainer` — backward-chain justifications |
| **User Interface** | FastAPI `/predict` endpoint → JSON response with full reasoning trace |

#### 5.4 Expert System Life Cycle
This system followed the Expert System Development Life Cycle:
1. **Knowledge Acquisition**: domain knowledge captured from phishing research literature (SUSPICIOUS_WORDS, BRAND_NAMES, SUSPICIOUS_TLDS, LEGIT_CHAR_BIGRAMS, rule weights)
2. **Knowledge Representation**: encoded as Python lambdas over feature dicts (Horn clauses)
3. **Prototype**: initial single Random Forest model
4. **Evaluation**: four separate classifiers evaluated + ensemble selected
5. **Refinement**: NLP/HMM features added; 6 new CSP constraints added; 9 new FOL rules added
6. **Deployment**: FastAPI REST service

---

## Model Zoo

| File | Algorithm | Syllabus Concept | Test F1 |
|---|---|---|---|
| `phishing_model_rf.joblib` | Random Forest (300 trees) | Unit 1 — Search (DFS tree ensemble) | 0.8826 |
| `phishing_model_nb.joblib` | Gaussian Naive Bayes | Unit 4 — Bayesian Network | 0.5765 |
| `phishing_model_gb.joblib` | Gradient Boosting (200 stages) | Unit 1 — Greedy Best-First | 0.8763 |
| `phishing_model_dt.joblib` | Decision Tree (depth≤12) | Unit 2/3 — Logical rules / CSP | 0.8131 |
| `phishing_model_ensemble.joblib` | Soft-Voting (all 4) | Unit 5 — Expert System | 0.8492 |

The **Ensemble** is used as the primary model by the API.

---

## Feature Set (29 Features)

### Structural / Lexical (23 features)

| Feature | Description |
|---|---|
| `url_length` | Total URL character length |
| `hostname_length` | Length of hostname |
| `has_ip` | Hostname is a raw IP address |
| `dot_count` | Number of dots |
| `hyphen_count` | Number of hyphens |
| `at_count` | Number of '@' characters |
| `question_count` | Number of '?' |
| `and_count` | Number of '&' |
| `equal_count` | Number of '=' |
| `underscore_count` | Number of '_' |
| `percent_count` | Number of '%' (URL encoding) |
| `slash_count` | Number of '/' |
| `digit_count` | Total digit characters |
| `digit_ratio_url` | `digit_count / url_length` |
| `has_https` | URL starts with https:// |
| `has_http_in_path` | HTTP embedded in path / open redirect |
| `has_double_slash` | '//' appears in path after scheme |
| `tld_in_path` | TLD string appears in URL path |
| `subdomain_count` | Number of subdomain levels |
| `contains_punycode` | 'xn--' in hostname (homograph attack) |
| `has_port` | Non-standard port specified |
| `suspicious_words` | Count of phishing-indicator tokens |
| `shortening_service` | Domain is a URL shortener |

### NLP / Entropy / HMM-Inspired (6 features)

| Feature | Concept | Description |
|---|---|---|
| `entropy` | Information Theory (Unit 4) | Shannon entropy over character distribution |
| `vowel_ratio` | NLP Tokenisation (Unit 5) | Vowel fraction of alphabetic characters |
| `avg_token_length` | NLP Tokenisation (Unit 5) | Mean alphanumeric token length |
| `bigram_anomaly` | HMM Emissions (Unit 4) | Fraction of alpha bigrams outside common legitimate set |
| `has_suspicious_tld` | Knowledge Base (Unit 3) | TLD in high-risk registrar list |
| `brand_impersonation` | NLP Pattern Match (Unit 5) | Brand name in subdomain or path |

---

## API Endpoints

### `POST /predict`
Analyse a URL. Returns ML prediction + full AI reasoning trace.

**Request body:**
```json
{ "url": "http://paypal-login.xyz/secure/verify" }
```

**Response structure:**
```json
{
  "label":          "phishing",
  "ml_probability": 0.91,
  "model_used":     "ensemble",
  "url_features":   { … },
  "rule_engine": {
    "rules_fired_count": 5,
    "rule_score": 0.52,
    "rules_fired": [
      { "id": "R01", "name": "IP Address Host", "fol": "…", "weight": 3 }
    ]
  },
  "csp_analysis": {
    "violations_count": 4,
    "satisfaction_score": 0.73,
    "violations": [ { "id": "C02", "constraint": "no_ip_host", … } ]
  },
  "bayesian_score":  0.87,
  "graph_search": {
    "total_nodes": 8,
    "suspicious_nodes": 2,
    "bfs_order": ["ROOT", "scheme:http", …],
    "a_star_top5": [[2.0, "path:verify"], …]
  },
  "explanations": [
    "[R18] Suspicious TLD: SuspiciousTLD(url) → HighRiskRegistrar(url) (weight=2)",
    "[R19] Brand Impersonation: BrandInSubdomainOrPath(url) → Impersonation(url) (weight=3)"
  ]
}
```

### `GET /analyze?url=<url>`
GET-friendly alias for `/predict`.

### `GET /model-info`
Returns full metadata: feature list, per-model metrics, feature importances, AI concepts map.

### `GET /health`
```json
{ "status": "ok", "model": "ensemble" }
```

---

## Files

```
backend/
  features.py         — Feature extraction (29 features, NLP + structural)
  ai_agent.py         — All AI reasoning modules (Agent, Search, CSP, Logic, Bayes, NLP)
  train_model.py      — Multi-model training pipeline
  app.py              — FastAPI REST API
  requirements.txt    — Python dependencies
  model/
    phishing_model_rf.joblib       — Random Forest
    phishing_model_nb.joblib       — Naive Bayes
    phishing_model_gb.joblib       — Gradient Boosting
    phishing_model_dt.joblib       — Decision Tree
    phishing_model_ensemble.joblib — Soft-Voting Ensemble (primary)
    metadata.json                  — Metrics, features, AI concept map
```

---

## Requirements

```
fastapi
uvicorn
pandas
scikit-learn
joblib
```

