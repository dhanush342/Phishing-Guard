"""
ai_agent.py
===========
AI reasoning engine for the Phishing Detection System.

Implements the following concepts from the AI syllabus:

  Unit 1 – Intelligent Agents
    • PhishingDetectionAgent   : PEAS-based rational agent
    • URLComponentGraph        : URL modelled as a directed graph
    • BFS, DFS, UCS            : Uninformed search over URL component graph
    • GreedyBestFirst, A*      : Informed (heuristic) search strategies

  Unit 2 – Logical Agents & Reasoning
    • ForwardChainRuleEngine   : Expert System forward-chaining inference
    • BackwardChainExplainer   : Backward-chaining explanation facility
    • PHISHING_RULES           : Knowledge base (propositional / FOL rules)

  Unit 3 – Knowledge Representation & Reasoning
    • UrlConstraintSatisfaction: CSP — Variables, Domains, Constraints
    • PhishingKnowledgeBase    : Structured fact + rule storage

  Unit 4 – Uncertainty & Probabilistic Reasoning
    • BayesianURLReasoner      : Naive Bayesian Network  P(Phishing | features)

  Unit 5 – NLP & Expert Systems
    • NLPUrlFeatureExtractor   : Shannon entropy, bigram HMM scoring, tokenisation
    • Expert System architecture: knowledge base + inference engine + explanation
"""

import math
import heapq
from collections import deque
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 1 — INTELLIGENT AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

class PhishingDetectionAgent:
    """
    Rational Agent for Phishing URL Detection (PEAS Framework).

    Performance Measure : Accuracy, F1-score, minimal false-negative rate
    Environment         : Internet URL space
                          (partially observable, stochastic, episodic,
                           static, discrete, single-agent)
    Actuators           : Phishing classification label + probability +
                          natural-language explanation
    Sensors             : Raw URL string → structured feature vector

    Agent type: Simple Reflex + Model-based (uses ML model as internal state).
    The agent perceives a URL, extracts features, applies its reasoning
    modules, and acts by returning a structured decision.
    """

    def __init__(self):
        self.rule_engine   = ForwardChainRuleEngine()
        self.csp           = UrlConstraintSatisfaction()
        self.bayesian      = BayesianURLReasoner()
        self.nlp_extractor = NLPUrlFeatureExtractor()
        self.explainer     = BackwardChainExplainer()

    def perceive(self, url: str) -> Dict[str, Any]:
        """Sensor: perceive the environment (raw URL) and build percept state."""
        from features import extract_url_features
        return extract_url_features(url)

    def act(self, url: str) -> Dict[str, Any]:
        """
        Full agent action cycle:
          1. Perceive (sensor)
          2. Forward-chain rule inference
          3. CSP constraint evaluation
          4. Bayesian probability estimation
          5. URL graph search analysis
          6. Backward-chain explanation
        Returns the complete reasoning trace.
        """
        percept              = self.perceive(url)
        rules_fired, r_score = self.rule_engine.run(percept)
        violations           = self.csp.check_violations(percept)
        csp_score            = self.csp.satisfaction_score(percept)
        bayes_score          = self.bayesian.score(percept)
        explanations         = self.explainer.explain(percept)
        graph                = URLComponentGraph(url)
        nlp                  = self.nlp_extractor.extract(url)

        return {
            "percept":          percept,
            "nlp_features":     nlp,
            "rules_fired":      rules_fired,
            "rule_score":       r_score,
            "csp_violations":   violations,
            "csp_score":        round(csp_score, 4),
            "bayesian_score":   bayes_score,
            "explanations":     explanations,
            "graph_summary": {
                "nodes":              len(graph.nodes),
                "bfs_order":          graph.bfs()[:8],
                "dfs_order":          graph.dfs()[:8],
                "a_star_top5":        [(round(f, 2), n)
                                       for f, n in graph.a_star()[:5]],
                "suspicious_nodes":   graph.suspicious_node_count(),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 1 — SEARCH STRATEGIES (URL Component Graph)
# ═══════════════════════════════════════════════════════════════════════════════

class URLComponentGraph:
    """
    Models a URL as a directed acyclic graph:
      ROOT → scheme → host-parts (subdomain → domain → tld)
           → path-segments → query-params

    Enables uninformed (BFS, DFS, UCS) and informed (Greedy, A*)
    search over the URL structural components to detect suspicious patterns.
    """

    _SUSPICIOUS_LABELS = {
        "login", "secure", "verify", "update", "account",
        "bank", "auth", "confirm", "password", "billing",
    }

    def __init__(self, url: str):
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        self.url = url
        self.nodes: List[str] = []
        self.edges: Dict[str, List[str]] = {}
        self._build_graph(url)

    def _build_graph(self, url: str) -> None:
        parsed = urlparse(url)
        host   = parsed.hostname or ""

        def add_edge(parent: str, child: str) -> None:
            self.nodes.append(child)
            self.edges.setdefault(parent, []).append(child)
            self.edges.setdefault(child, [])

        self.nodes = ["ROOT"]
        self.edges = {"ROOT": []}

        scheme_node = f"scheme:{parsed.scheme}"
        add_edge("ROOT", scheme_node)

        prev = scheme_node
        for part in (host.split(".") if host else []):
            n = f"host:{part}"
            add_edge(prev, n)
            prev = n

        for seg in parsed.path.split("/"):
            if seg:
                n = f"path:{seg[:30]}"
                add_edge(prev, n)
                prev = n

        if parsed.query:
            for param in parsed.query.split("&"):
                n = f"query:{param[:20]}"
                add_edge(prev, n)

    # ── Uninformed Search ────────────────────────────────────────────────────

    def bfs(self, start: str = "ROOT") -> List[str]:
        """Breadth-First Search — explores nodes level by level."""
        visited, order, queue = set(), [], deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                order.append(node)
                for nb in self.edges.get(node, []):
                    if nb not in visited:
                        queue.append(nb)
        return order

    def dfs(self, start: str = "ROOT") -> List[str]:
        """Depth-First Search — explores deepest paths first."""
        visited, order, stack = set(), [], [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                order.append(node)
                for nb in reversed(self.edges.get(node, [])):
                    if nb not in visited:
                        stack.append(nb)
        return order

    def ucs(self, start: str = "ROOT") -> List[Tuple[float, str]]:
        """
        Uniform-Cost Search — expands node with lowest cumulative path cost.
        Cost g(n) = number of edges from ROOT to n.
        """
        heap:    List[Tuple[float, str]] = [(0.0, start)]
        visited: Dict[str, float]        = {}
        order:   List[Tuple[float, str]] = []
        while heap:
            cost, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited[node] = cost
            order.append((cost, node))
            for nb in self.edges.get(node, []):
                if nb not in visited:
                    heapq.heappush(heap, (cost + 1.0, nb))
        return order

    # ── Informed (Heuristic) Search ──────────────────────────────────────────

    @staticmethod
    def _heuristic(node: str) -> float:
        """
        Admissible heuristic h(n): counts suspicious keywords inside
        the node label.  Used by Greedy Best-First and A*.
        """
        label = node.split(":", 1)[-1].lower()
        return float(sum(
            1 for w in URLComponentGraph._SUSPICIOUS_LABELS if w in label
        ))

    def greedy_best_first(self, start: str = "ROOT") -> List[str]:
        """
        Greedy Best-First Search — always expands the node that looks closest
        to the goal (most suspicious) using h(n) alone.
        """
        heap    = [(self._heuristic(start), start)]
        visited = set()
        order   = []
        while heap:
            _, node = heapq.heappop(heap)
            if node not in visited:
                visited.add(node)
                order.append(node)
                for nb in self.edges.get(node, []):
                    if nb not in visited:
                        heapq.heappush(heap, (self._heuristic(nb), nb))
        return order

    def a_star(self, start: str = "ROOT") -> List[Tuple[float, str]]:
        """
        A* Search — f(n) = g(n) + h(n).
        g(n) = actual path cost from start.
        h(n) = heuristic suspicion estimate.
        Guarantees optimal ordering when h is admissible.
        """
        heap    = [(self._heuristic(start), 0.0, start)]
        visited: Dict[str, float] = {}
        order:   List[Tuple[float, str]] = []
        while heap:
            f, g, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited[node] = g
            order.append((f, node))
            for nb in self.edges.get(node, []):
                if nb not in visited:
                    g2 = g + 1.0
                    heapq.heappush(heap, (g2 + self._heuristic(nb), g2, nb))
        return order

    def suspicious_node_count(self) -> int:
        """Number of nodes with non-zero heuristic suspicion score."""
        return sum(1 for _, n in self.a_star() if self._heuristic(n) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 3 — CONSTRAINT SATISFACTION PROBLEM (CSP)
# ═══════════════════════════════════════════════════════════════════════════════

class UrlConstraintSatisfaction:
    """
    URL Legitimacy as a Constraint Satisfaction Problem.

    Variables  : URL structural attributes (one per feature dimension)
    Domains    : Acceptable value ranges inferred from legitimate-URL knowledge
    Constraints: Logical conditions that must hold for a URL to be legitimate

    Each violated constraint contributes evidence of phishing.
    This directly implements CSP arc-consistency style checking.
    """

    # Each constraint: (id, name, test_fn, natural-language description)
    CONSTRAINTS: List[Tuple[str, str, Any, str]] = [
        ("C01", "length_ok",
         lambda f: f["url_length"] <= 75,
         "URL length ≤ 75 chars"),
        ("C02", "no_ip_host",
         lambda f: f["has_ip"] == 0,
         "Hostname must not be a raw IP address"),
        ("C03", "uses_https",
         lambda f: f["has_https"] == 1,
         "URL must use HTTPS scheme"),
        ("C04", "no_at_sign",
         lambda f: f["at_count"] == 0,
         "No '@' character allowed in URL"),
        ("C05", "no_http_in_path",
         lambda f: f["has_http_in_path"] == 0,
         "No embedded HTTP in the URL path"),
        ("C06", "no_double_slash",
         lambda f: f["has_double_slash"] == 0,
         "No '//' in path after scheme"),
        ("C07", "low_subdomain_depth",
         lambda f: f["subdomain_count"] <= 2,
         "Subdomain depth ≤ 2"),
        ("C08", "no_punycode",
         lambda f: f["contains_punycode"] == 0,
         "No punycode (xn--) in hostname"),
        ("C09", "no_shortener",
         lambda f: f["shortening_service"] == 0,
         "Not a URL-shortening service"),
        ("C10", "low_suspicion_tokens",
         lambda f: f["suspicious_words"] <= 1,
         "≤ 1 suspicious keyword in URL"),
        ("C11", "low_digit_ratio",
         lambda f: f["digit_ratio_url"] <= 0.35,
         "Digit ratio ≤ 35 %"),
        ("C12", "low_entropy",
         lambda f: f["entropy"] <= 4.5,
         "Shannon entropy ≤ 4.5 bits"),
        ("C13", "normal_bigrams",
         lambda f: f["bigram_anomaly"] <= 0.75,
         "Bigram anomaly score ≤ 0.75"),
        ("C14", "safe_tld",
         lambda f: f["has_suspicious_tld"] == 0,
         "TLD is not in high-risk list"),
        ("C15", "no_brand_impersonation",
         lambda f: f["brand_impersonation"] == 0,
         "No brand name impersonation detected"),
    ]

    def check_violations(self, features: Dict[str, Any]) -> List[Dict[str, str]]:
        """Return list of violated constraints (arc-inconsistency evidence)."""
        violations = []
        for cid, name, test, desc in self.CONSTRAINTS:
            try:
                if not test(features):
                    violations.append({"id": cid, "constraint": name,
                                       "description": desc})
            except Exception:
                pass
        return violations

    def satisfaction_score(self, features: Dict[str, Any]) -> float:
        """Fraction of constraints satisfied (1.0 = fully legitimate pattern)."""
        total     = len(self.CONSTRAINTS)
        satisfied = total - len(self.check_violations(features))
        return satisfied / total


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 2 — LOGICAL AGENTS: Knowledge Base (Propositional / FOL rules)
# ═══════════════════════════════════════════════════════════════════════════════

# Each rule is analogous to a Horn clause in propositional logic:
#   HEAD  ← BODY
#   Phishing(url) ← Condition(features)
#
# The FOL notation in "description" captures the first-order interpretation.

PHISHING_RULES: List[Dict[str, Any]] = [
    {
        "id": "R01", "name": "IP Address Host", "weight": 3,
        "antecedent": lambda f: f["has_ip"] == 1,
        "description": "HasIPAddress(url) → SuspectHost(url)",
    },
    {
        "id": "R02", "name": "No HTTPS", "weight": 2,
        "antecedent": lambda f: f["has_https"] == 0,
        "description": "¬UsesHTTPS(url) → InsecureChannel(url)",
    },
    {
        "id": "R03", "name": "At-Sign in URL", "weight": 3,
        "antecedent": lambda f: f["at_count"] > 0,
        "description": "ContainsAt(url) → CredentialHiding(url)",
    },
    {
        "id": "R04", "name": "Embedded HTTP in Path", "weight": 2,
        "antecedent": lambda f: f["has_http_in_path"] == 1,
        "description": "HTTPInPath(url) → OpenRedirect(url)",
    },
    {
        "id": "R05", "name": "URL Shortening Service", "weight": 2,
        "antecedent": lambda f: f["shortening_service"] == 1,
        "description": "ShorteningService(url) → ObfuscatedDestination(url)",
    },
    {
        "id": "R06", "name": "High Suspicious Token Count", "weight": 3,
        "antecedent": lambda f: f["suspicious_words"] >= 2,
        "description": "HighSuspicionTokens(url,N) ∧ N≥2 → SocialEngineering(url)",
    },
    {
        "id": "R07", "name": "Punycode Hostname", "weight": 3,
        "antecedent": lambda f: f["contains_punycode"] == 1,
        "description": "PunycodeHost(url) → HomographAttack(url)",
    },
    {
        "id": "R08", "name": "Excessive Subdomains", "weight": 2,
        "antecedent": lambda f: f["subdomain_count"] > 3,
        "description": "ExcessiveSubdomains(url,N) ∧ N>3 → DomainObfuscation(url)",
    },
    {
        "id": "R09", "name": "Very Long URL", "weight": 1,
        "antecedent": lambda f: f["url_length"] > 100,
        "description": "LongURL(url) ∧ Length>100 → PotentialObfuscation(url)",
    },
    {
        "id": "R10", "name": "Double Slash in Path", "weight": 1,
        "antecedent": lambda f: f["has_double_slash"] == 1,
        "description": "DoubleSlashPath(url) → PathManipulation(url)",
    },
    {
        "id": "R11", "name": "IP + No HTTPS (Combined Rule)", "weight": 2,
        "antecedent": lambda f: f["has_ip"] == 1 and f["has_https"] == 0,
        "description": "HasIPAddress(url) ∧ ¬UsesHTTPS(url) → HighRiskHost(url)",
    },
    {
        "id": "R12", "name": "High Digit Ratio", "weight": 1,
        "antecedent": lambda f: f["digit_ratio_url"] > 0.35,
        "description": "HighDigitRatio(url) → ObfuscatedIdentity(url)",
    },
    {
        "id": "R13", "name": "Non-Standard Port", "weight": 1,
        "antecedent": lambda f: f["has_port"] == 1,
        "description": "NonStandardPort(url) → AtypicalService(url)",
    },
    {
        "id": "R14", "name": "TLD Appears in Path", "weight": 2,
        "antecedent": lambda f: f["tld_in_path"] == 1,
        "description": "TLDInPath(url) → DomainSpoofing(url)",
    },
    {
        "id": "R15", "name": "Many Query Parameters", "weight": 1,
        "antecedent": lambda f: f["and_count"] > 3,
        "description": "ManyQueryParams(url,N) ∧ N>3 → DataHarvesting(url)",
    },
    {
        "id": "R16", "name": "High Shannon Entropy", "weight": 2,
        "antecedent": lambda f: f["entropy"] > 4.5,
        "description": "HighEntropy(url) → RandomisedSubdomain(url)",
    },
    {
        "id": "R17", "name": "High Bigram Anomaly", "weight": 2,
        "antecedent": lambda f: f["bigram_anomaly"] > 0.75,
        "description": "HighBigramAnomaly(url) → AutoGeneratedDomain(url)",
    },
    {
        "id": "R18", "name": "Suspicious TLD", "weight": 2,
        "antecedent": lambda f: f["has_suspicious_tld"] == 1,
        "description": "SuspiciousTLD(url) → HighRiskRegistrar(url)",
    },
    {
        "id": "R19", "name": "Brand Impersonation", "weight": 3,
        "antecedent": lambda f: f["brand_impersonation"] == 1,
        "description": "BrandInSubdomainOrPath(url) → Impersonation(url)",
    },
]

_MAX_RULE_SCORE = float(sum(r["weight"] for r in PHISHING_RULES))


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 2 — FORWARD CHAINING (Modus Ponens inference engine)
# ═══════════════════════════════════════════════════════════════════════════════

class ForwardChainRuleEngine:
    """
    Forward Chaining Inference Engine.

    Algorithm (data-driven, bottom-up):
      1. Load all URL feature values as facts into working memory.
      2. For each rule in the knowledge base:
           IF antecedent(facts) is TRUE  →  fire rule, record consequence.
      3. Accumulate weighted evidence score; normalise to [0, 1].

    This is the core inference engine of the Expert System (Unit 5).
    Rules are expressed as propositional Horn clauses.
    """

    def run(self, features: Dict[str, Any]
            ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Returns (fired_rules, normalised_score).
        fired_rules: list of rule dicts that triggered.
        normalised_score: cumulative risk in [0, 1].
        """
        fired: List[Dict[str, Any]] = []
        total_score = 0.0
        for rule in PHISHING_RULES:
            try:
                if rule["antecedent"](features):
                    fired.append({
                        "id":          rule["id"],
                        "name":        rule["name"],
                        "weight":      rule["weight"],
                        "fol":         rule["description"],
                    })
                    total_score += rule["weight"]
            except Exception:
                pass
        normalised = total_score / _MAX_RULE_SCORE if _MAX_RULE_SCORE > 0 else 0.0
        return fired, round(normalised, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 2 — BACKWARD CHAINING (goal-driven explanation)
# ═══════════════════════════════════════════════════════════════════════════════

class BackwardChainExplainer:
    """
    Backward Chaining Explainer.

    Given the goal "URL is phishing", traces backwards through the
    rule base to enumerate which rules support that conclusion and
    provides human-readable explanations.

    This implements the Explanation Facility of the Expert System (Unit 5).
    Algorithm (goal-driven, top-down):
      1. Start with goal: Phishing(url).
      2. Find all rules whose consequent supports Phishing.
      3. For each rule: check whether its antecedent holds (reduce to sub-goals).
      4. Return natural-language justifications for all confirmed chains.
    """

    def explain(self, features: Dict[str, Any],
                goal: str = "phishing") -> List[str]:
        """
        Returns ordered list of natural-language rule justifications.
        """
        engine = ForwardChainRuleEngine()
        fired, _ = engine.run(features)
        if not fired:
            return ["No phishing evidence found via logical backward chaining."]
        return [
            f"[{r['id']}] {r['name']}: {r['fol']} (weight={r['weight']})"
            for r in fired
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 3 — KNOWLEDGE REPRESENTATION: Phishing Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

class PhishingKnowledgeBase:
    """
    Structured Knowledge Base for the Expert System.

    Stores:
      - facts    : observed feature values for a URL (working memory)
      - rules    : PHISHING_RULES (long-term memory)
      - metadata : derived conclusions (inferred facts)

    This mirrors the Knowledge Representation & Reasoning module (Unit 3):
    planning with state-space search where state = set of known facts.
    """

    def __init__(self):
        self.facts:    Dict[str, Any]       = {}
        self.inferred: List[str]            = []
        self.rules:    List[Dict[str, Any]] = PHISHING_RULES

    def load_facts(self, features: Dict[str, Any]) -> None:
        """Populate working memory with perceived feature values."""
        self.facts = dict(features)
        self.inferred = []

    def infer(self) -> List[str]:
        """Run forward chaining and store inferred conclusions."""
        engine = ForwardChainRuleEngine()
        fired, _ = engine.run(self.facts)
        self.inferred = [r["description"] for r in fired]
        return self.inferred


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 4 — PROBABILISTIC REASONING: Bayesian URL Reasoner
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianURLReasoner:
    """
    Naive Bayesian Network for Phishing Probability Estimation.

    Models P(Phishing | f_1, f_2, …, f_n) under the conditional
    independence assumption (Naive Bayes):

        P(Phishing | F) ∝ P(Phishing) · ∏ P(f_i | Phishing)

    Each feature node is a Bayesian Network variable connected only to
    the latent class node (Phishing / Legitimate).
    Conditional probabilities are estimated from domain knowledge.

    Inference is performed via log-space Bayes' theorem to avoid
    numerical underflow (analogous to the Variable Elimination algorithm
    in Bayesian Networks).
    """

    PRIOR_PHISHING = 0.50   # P(Phishing) ≈ 0.5 from balanced dataset

    # P(feature_observed | class) — domain-knowledge-based CPTs
    #   key  : feature name
    #   value: (P(feat=1|phishing), P(feat=1|legitimate))
    CPT: Dict[str, Tuple[float, float]] = {
        "has_ip":              (0.38,  0.01),
        "has_https":           (0.18,  0.80),   # low HTTPS in phishing
        "at_count":            (0.14,  0.001),
        "has_http_in_path":    (0.22,  0.02),
        "shortening_service":  (0.16,  0.03),
        "contains_punycode":   (0.08,  0.005),
        "has_double_slash":    (0.12,  0.02),
        "has_port":            (0.11,  0.04),
        "tld_in_path":         (0.25,  0.05),
        "has_suspicious_tld":  (0.30,  0.04),
        "brand_impersonation": (0.28,  0.02),
    }

    def score(self, features: Dict[str, Any]) -> float:
        """
        Compute posterior P(Phishing | features) using Naive Bayes.
        Returns probability in [0, 1].
        """
        log_p = math.log(self.PRIOR_PHISHING + 1e-9)         # log P(phi)
        log_q = math.log(1.0 - self.PRIOR_PHISHING + 1e-9)   # log P(leg)

        for feat, (p_phi, p_leg) in self.CPT.items():
            raw = features.get(feat, 0)
            # For has_https: it is ABSENCE that is suspicious
            obs = (1 - int(bool(raw))) if feat == "has_https" else int(bool(raw))

            if obs == 1:
                log_p += math.log(p_phi + 1e-9)
                log_q += math.log(p_leg + 1e-9)
            else:
                log_p += math.log(1.0 - p_phi + 1e-9)
                log_q += math.log(1.0 - p_leg + 1e-9)

        # Normalise via log-sum-exp for numerical stability
        max_log = max(log_p, log_q)
        p_norm  = math.exp(log_p - max_log)
        q_norm  = math.exp(log_q - max_log)
        return round(p_norm / (p_norm + q_norm + 1e-9), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT 5 — NLP & HMM-INSPIRED URL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

class NLPUrlFeatureExtractor:
    """
    NLP-inspired feature extractor (Unit 5 — Natural Language Processing).

    Treats a URL as a character / token sequence and computes:
      • Shannon Entropy     : randomness measure (NLP information theory)
      • Vowel Ratio         : linguistic normality indicator
      • Average Token Length: NLP tokenisation metric
      • Bigram Anomaly Score: HMM-inspired character bigram transition metric
                              (phishing domains often have auto-generated,
                               random character sequences with unusual bigrams)

    The bigram model acts as a simplified HMM emission model:
        P(char_t | char_{t-1}) — high anomaly ↔ low emission probability
        in the legitimate-URL Markov chain.
    """

    # Common character bigrams in legitimate English-based URLs
    LEGIT_BIGRAMS = {
        "ww", "co", "ht", "tp", "er", "en", "in", "on", "an",
        "re", "ou", "at", "ed", "it", "es", "or", "to", "is",
        "te", "st", "ng", "ar", "ti", "al", "le", "om", "ne",
        "se", "ha", "ve", "de", "ro", "li", "si", "nd", "ra",
    }

    def extract(self, url: str) -> Dict[str, float]:
        """Return dict of NLP features for the given URL."""
        norm = url if url.startswith(("http://", "https://")) else "http://" + url
        return {
            "entropy":          round(self._entropy(norm), 4),
            "vowel_ratio":      round(self._vowel_ratio(norm), 4),
            "avg_token_length": round(self._avg_token_len(norm), 4),
            "bigram_anomaly":   round(self._bigram_anomaly(norm), 4),
        }

    @staticmethod
    def _entropy(text: str) -> float:
        """Shannon entropy H = -∑ p(c) log₂ p(c) over character distribution."""
        if not text:
            return 0.0
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in freq.values())

    @staticmethod
    def _vowel_ratio(text: str) -> float:
        alpha = [c for c in text.lower() if c.isalpha()]
        if not alpha:
            return 0.0
        return sum(1 for c in alpha if c in "aeiou") / len(alpha)

    @staticmethod
    def _avg_token_len(url: str) -> float:
        import re
        tokens = re.findall(r"[a-zA-Z0-9]+", url)
        if not tokens:
            return 0.0
        return sum(len(t) for t in tokens) / len(tokens)

    def _bigram_anomaly(self, text: str) -> float:
        bigrams = [
            text[i: i + 2].lower()
            for i in range(len(text) - 1)
            if text[i].isalpha() and text[i + 1].isalpha()
        ]
        if not bigrams:
            return 0.0
        anomalous = sum(1 for bg in bigrams if bg not in self.LEGIT_BIGRAMS)
        return anomalous / len(bigrams)


# ═══════════════════════════════════════════════════════════════════════════════
# Public convenience API
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_url(url: str) -> Dict[str, Any]:
    """
    Run the full AI agent pipeline on a URL.
    Returns complete reasoning trace from all modules.
    """
    agent = PhishingDetectionAgent()
    return agent.act(url)
