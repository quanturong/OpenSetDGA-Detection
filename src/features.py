"""
Lexical feature extraction for domain names.
Produces a numeric feature vector from the raw domain string.
"""

import math
import re
from collections import Counter

import numpy as np
import tldextract


# ── character sets ──────────────────────────────────────────────────────────
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
HEX_CHARS = set("0123456789abcdef")

# ── n-gram frequency tables (English bigram freq, precomputed on Alexa-1M) ─
# We approximate "normality" via entropy of char bigrams instead of a
# heavyweight lookup table – keeps the module self-contained.

# ── helpers ─────────────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    """Shannon entropy of character distribution in *s*."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _ngram_entropy(s: str, n: int) -> float:
    """Shannon entropy over character n-grams."""
    if len(s) < n:
        return 0.0
    ngrams = [s[i:i + n] for i in range(len(s) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _longest_run(s: str, charset: set) -> int:
    """Length of the longest consecutive run of characters in *charset*."""
    best = cur = 0
    for ch in s:
        if ch in charset:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _consecutive_consonant_ratio(s: str) -> float:
    """Ratio of the longest consonant run to string length."""
    if not s:
        return 0.0
    return _longest_run(s, CONSONANTS) / len(s)


def _consecutive_digit_ratio(s: str) -> float:
    if not s:
        return 0.0
    return _longest_run(s, set("0123456789")) / len(s)


def _repeated_char_ratio(s: str) -> float:
    """Fraction of characters that appear more than once."""
    if not s:
        return 0.0
    counts = Counter(s)
    return sum(1 for c in counts.values() if c > 1) / len(counts)


def _gini_index(s: str) -> float:
    """Gini impurity of character distribution."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


# ── main extraction ────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "length",
    "sld_length",
    "n_labels",
    "digit_count",
    "digit_ratio",
    "alpha_count",
    "alpha_ratio",
    "hyphen_count",
    "hyphen_ratio",
    "vowel_count",
    "vowel_ratio",
    "consonant_count",
    "consonant_ratio",
    "unique_chars",
    "unique_char_ratio",
    "hex_char_ratio",
    "char_entropy",
    "bigram_entropy",
    "trigram_entropy",
    "longest_consonant_run",
    "longest_digit_run",
    "consec_consonant_ratio",
    "consec_digit_ratio",
    "repeated_char_ratio",
    "gini_index",
    "has_digits",
    "starts_with_digit",
    "digit_alpha_transitions",
    "max_label_length",
    "mean_label_length",
    "std_label_length",
    "tld_is_common",
    "sld_digit_ratio",
    "sld_entropy",
    "subdomain_count",
]


# Top TLDs considered "common" (Tranco-derived)
_COMMON_TLDS = frozenset([
    "com", "net", "org", "de", "uk", "ru", "br", "au", "cn", "fr",
    "it", "nl", "pl", "ca", "es", "in", "jp", "info", "eu", "co",
    "io", "me", "tv", "cc", "biz", "us", "xyz", "online", "site",
    "top", "edu", "gov",
])


def extract_features_single(domain: str) -> np.ndarray:
    """Return a 1-D float32 feature array for one domain string."""
    domain_lower = domain.lower().strip().rstrip(".")

    ext = tldextract.extract(domain_lower)
    sld = ext.domain          # second-level domain (no TLD)
    suffix = ext.suffix       # TLD / public suffix
    subdomain = ext.subdomain

    labels = [p for p in domain_lower.split(".") if p]
    n_labels = len(labels)

    full = domain_lower.replace(".", "")  # all chars without dots
    full_len = len(full) or 1

    sld = sld or full
    sld_len = len(sld) or 1

    digit_count = sum(c.isdigit() for c in full)
    alpha_count = sum(c.isalpha() for c in full)
    hyphen_count = full.count("-")
    vowel_count = sum(c in VOWELS for c in full)
    consonant_count = sum(c in CONSONANTS for c in full)
    unique_chars = len(set(full))
    hex_count = sum(c in HEX_CHARS for c in full)

    label_lengths = [len(l) for l in labels] if labels else [0]

    transitions = 0
    for i in range(1, len(full)):
        if full[i].isdigit() != full[i - 1].isdigit():
            transitions += 1

    sub_parts = [p for p in subdomain.split(".") if p]

    feats = np.array([
        len(domain_lower),                          # length
        len(sld),                                    # sld_length
        n_labels,                                    # n_labels
        digit_count,                                 # digit_count
        digit_count / full_len,                      # digit_ratio
        alpha_count,                                 # alpha_count
        alpha_count / full_len,                      # alpha_ratio
        hyphen_count,                                # hyphen_count
        hyphen_count / full_len,                     # hyphen_ratio
        vowel_count,                                 # vowel_count
        vowel_count / full_len,                      # vowel_ratio
        consonant_count,                             # consonant_count
        consonant_count / full_len,                  # consonant_ratio
        unique_chars,                                # unique_chars
        unique_chars / full_len,                     # unique_char_ratio
        hex_count / full_len,                        # hex_char_ratio
        _entropy(full),                              # char_entropy
        _ngram_entropy(full, 2),                     # bigram_entropy
        _ngram_entropy(full, 3),                     # trigram_entropy
        _longest_run(full, CONSONANTS),              # longest_consonant_run
        _longest_run(full, set("0123456789")),       # longest_digit_run
        _consecutive_consonant_ratio(full),          # consec_consonant_ratio
        _consecutive_digit_ratio(full),              # consec_digit_ratio
        _repeated_char_ratio(full),                  # repeated_char_ratio
        _gini_index(full),                           # gini_index
        int(digit_count > 0),                        # has_digits
        int(full[0].isdigit()) if full else 0,       # starts_with_digit
        transitions,                                 # digit_alpha_transitions
        max(label_lengths),                          # max_label_length
        np.mean(label_lengths),                      # mean_label_length
        np.std(label_lengths),                       # std_label_length
        int(suffix in _COMMON_TLDS),                 # tld_is_common
        sum(c.isdigit() for c in sld) / sld_len,    # sld_digit_ratio
        _entropy(sld),                               # sld_entropy
        len(sub_parts),                              # subdomain_count
    ], dtype=np.float32)

    return feats


def extract_features_batch(domains: list[str]) -> np.ndarray:
    """Return (N, D) float32 feature matrix for a list of domains."""
    return np.vstack([extract_features_single(d) for d in domains])


# ── quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_domains = ["google.com", "xjk38dh2kq.ru", "my-cool-site.co.uk"]
    feats = extract_features_batch(test_domains)
    print(f"Feature shape: {feats.shape}")
    print(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
    for i, d in enumerate(test_domains):
        print(f"\n{d}:")
        for name, val in zip(FEATURE_NAMES, feats[i]):
            print(f"  {name:30s} = {val:.4f}")
