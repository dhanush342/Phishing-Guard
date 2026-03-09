"""
features.py
===========
URL feature extraction for the Phishing Detection System.

Concepts incorporated (from AI syllabus):
  - NLP (Unit 5): Shannon entropy, character n-gram bigram anomaly, token-level
    analysis, vowel/consonant ratio — treating the URL as a text sequence.
  - Hidden Markov Models (Unit 4): Character bigram transition scoring inspired
    by HMM emission probability modelling over URL character sequences.
  - Knowledge Representation (Unit 3): Structured feature vocabulary that
    encodes domain knowledge as measurable URL attributes.
  - Logical Agents (Unit 2): SUSPICIOUS_TLDS and BRANDS knowledge bases used
    as first-order logic predicates over URL attributes.
"""

import math
import re
from urllib.parse import urlparse

# ── Knowledge Bases (First-Order Logic predicate domains) ────────────────────

SUSPICIOUS_WORDS = {
    "login", "verify", "update", "secure", "account", "bank",
    "signin", "confirm", "password", "auth", "billing", "support",
    "webscr", "paypal", "apple", "microsoft", "ebay", "amazon",
    "chase", "wellsfargo", "netflix", "facebook", "instagram",
}

SHORTENING_DOMAINS = {
    "bit.ly", "t.co", "tinyurl.com", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly", "rb.gy", "short.io",
}

# High-risk free/abusable TLDs (knowledge base for HasSuspiciousTLD predicate)
SUSPICIOUS_TLDS = {
    "xyz", "tk", "ml", "ga", "cf", "gq", "pw", "top", "club",
    "work", "date", "racing", "stream", "download", "win",
    "bid", "trade", "science", "accountant", "loan",
}

# Brand names used to detect impersonation in subdomains / path
BRAND_NAMES = {
    "paypal", "google", "apple", "microsoft", "amazon", "facebook",
    "instagram", "netflix", "ebay", "chase", "wellsfargo",
    "linkedin", "twitter", "yahoo", "dropbox", "outlook",
}

# Empirically common character bigrams in legitimate URLs (HMM emission model)
LEGIT_CHAR_BIGRAMS = {
    "ww", "co", "ht", "tp", "er", "en", "in", "on", "an",
    "re", "ou", "at", "ed", "it", "es", "or", "to", "is",
    "te", "st", "ng", "ar", "ti", "al", "le", "om", "ne",
}

# ── Feature registry ─────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Structural / lexical features
    "url_length",
    "hostname_length",
    "has_ip",
    "dot_count",
    "hyphen_count",
    "at_count",
    "question_count",
    "and_count",
    "equal_count",
    "underscore_count",
    "percent_count",
    "slash_count",
    "digit_count",
    "digit_ratio_url",
    "has_https",
    "has_http_in_path",
    "has_double_slash",
    "tld_in_path",
    "subdomain_count",
    "contains_punycode",
    "has_port",
    "suspicious_words",
    "shortening_service",
    # NLP / entropy features (Unit 5 — NLP; Unit 4 — HMM-inspired)
    "entropy",
    "vowel_ratio",
    "avg_token_length",
    "bigram_anomaly",
    "has_suspicious_tld",
    "brand_impersonation",
]


# ── NLP / Entropy helpers (Unit 5: NLP; Unit 4: HMM character sequences) ────

def _shannon_entropy(text: str) -> float:
    """
    Shannon Entropy  H(X) = -∑ p(x) log₂ p(x)
    High entropy URLs are more random → characteristic of phishing domains.
    """
    if not text:
        return 0.0
    freq: dict = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _vowel_ratio(text: str) -> float:
    """Ratio of vowels to total alphabetic characters (NLP token feature)."""
    alpha = [c for c in text.lower() if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c in "aeiou") / len(alpha)


def _avg_token_length(url: str) -> float:
    """Average length of alphanumeric tokens (NLP tokenization)."""
    tokens = re.findall(r"[a-zA-Z0-9]+", url)
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def _bigram_anomaly(text: str) -> float:
    """
    HMM-inspired bigram anomaly score.
    Computes fraction of alphabetic character bigrams that are NOT in the
    common legitimate bigram set.  High score → unusual/random character
    sequences typical of auto-generated phishing subdomains.
    """
    bigrams = [
        text[i: i + 2].lower()
        for i in range(len(text) - 1)
        if text[i].isalpha() and text[i + 1].isalpha()
    ]
    if not bigrams:
        return 0.0
    anomalous = sum(1 for bg in bigrams if bg not in LEGIT_CHAR_BIGRAMS)
    return anomalous / len(bigrams)


def _has_suspicious_tld(host_lower: str) -> int:
    """HasSuspiciousTLD(url) — FOL predicate over TLD knowledge base."""
    tld = host_lower.split(".")[-1] if "." in host_lower else ""
    return 1 if tld in SUSPICIOUS_TLDS else 0


def _brand_impersonation(host_lower: str, path: str) -> int:
    """
    BrandInSubdomain(url) ∨ BrandInPath(url) — FOL predicate.
    Brand name appearing outside the primary registered domain signals
    impersonation (e.g. paypal.attacker.com, attacker.com/apple/login).
    """
    parts = [p for p in host_lower.split(".") if p]
    if len(parts) >= 3:
        subdomains = ".".join(parts[:-2])
        if any(brand in subdomains for brand in BRAND_NAMES):
            return 1
    if any(brand in path.lower() for brand in BRAND_NAMES):
        return 1
    return 0


# ── Main feature extraction ───────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    normalized = url.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = "http://" + normalized
    return normalized


def extract_url_features(url: str) -> dict:
    """
    Extract all FEATURE_NAMES features from a raw URL string.
    Combines structural, lexical, NLP, and HMM-inspired attributes.
    """
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    host = parsed.hostname or ""
    path = parsed.path or ""

    url_lower = normalized.lower()
    host_lower = host.lower()

    digit_count = sum(ch.isdigit() for ch in normalized)
    url_length = len(normalized)

    tld = host_lower.split(".")[-1] if "." in host_lower else ""
    tld_in_path = 1 if tld and tld in path.lower() else 0

    subdomain_count = 0
    if host_lower:
        parts = [p for p in host_lower.split(".") if p]
        if len(parts) > 2:
            subdomain_count = len(parts) - 2

    suspicious_word_count = sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)

    return {
        # ── Structural / lexical ────────────────────────────────────────────
        "url_length":        url_length,
        "hostname_length":   len(host),
        "has_ip":            1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) else 0,
        "dot_count":         normalized.count("."),
        "hyphen_count":      normalized.count("-"),
        "at_count":          normalized.count("@"),
        "question_count":    normalized.count("?"),
        "and_count":         normalized.count("&"),
        "equal_count":       normalized.count("="),
        "underscore_count":  normalized.count("_"),
        "percent_count":     normalized.count("%"),
        "slash_count":       normalized.count("/"),
        "digit_count":       digit_count,
        "digit_ratio_url":   digit_count / max(1, url_length),
        "has_https":         1 if normalized.lower().startswith("https://") else 0,
        "has_http_in_path":  1 if "http" in path.lower() else 0,
        "has_double_slash":  1 if "//" in path else 0,
        "tld_in_path":       tld_in_path,
        "subdomain_count":   subdomain_count,
        "contains_punycode": 1 if "xn--" in host_lower else 0,
        "has_port":          1 if parsed.port else 0,
        "suspicious_words":  suspicious_word_count,
        "shortening_service": 1 if host_lower in SHORTENING_DOMAINS else 0,
        # ── NLP / Entropy / HMM-inspired ───────────────────────────────────
        "entropy":            round(_shannon_entropy(normalized), 4),
        "vowel_ratio":        round(_vowel_ratio(normalized), 4),
        "avg_token_length":   round(_avg_token_length(normalized), 4),
        "bigram_anomaly":     round(_bigram_anomaly(normalized), 4),
        "has_suspicious_tld": _has_suspicious_tld(host_lower),
        "brand_impersonation": _brand_impersonation(host_lower, path),
    }
