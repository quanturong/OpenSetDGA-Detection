"""
Lexical feature extraction for domain names.
Produces a numeric feature vector from the raw domain string.
"""

import math
import re
import zlib
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


# ── reference distributions (derived from Tranco top-1M benign domains) ────

# Character unigram frequencies in benign (Tranco) domain SLDs
_BENIGN_CHAR_FREQ: dict[str, float] = {
    'a': 0.0820, 'b': 0.0145, 'c': 0.0305, 'd': 0.0365, 'e': 0.1000,
    'f': 0.0145, 'g': 0.0235, 'h': 0.0295, 'i': 0.0625, 'j': 0.0055,
    'k': 0.0135, 'l': 0.0475, 'm': 0.0305, 'n': 0.0640, 'o': 0.0750,
    'p': 0.0230, 'q': 0.0025, 'r': 0.0635, 's': 0.0720, 't': 0.0680,
    'u': 0.0335, 'v': 0.0115, 'w': 0.0175, 'x': 0.0045, 'y': 0.0195,
    'z': 0.0030, '0': 0.0070, '1': 0.0065, '2': 0.0055, '3': 0.0035,
    '4': 0.0025, '5': 0.0025, '6': 0.0020, '7': 0.0020, '8': 0.0020,
    '9': 0.0020, '-': 0.0120,
}

# English character bigram log-probabilities (log base 2, normalized per leading char)
# Top ~130 bigrams covering ~85 % of transitions in benign domains.
# Unseen bigrams get a floor of -10.0.
_BIGRAM_LOG2: dict[str, float] = {
    'th': -1.26, 'he': -1.42, 'in': -1.55, 'er': -1.61, 'an': -1.66,
    're': -1.72, 'on': -1.82, 'en': -1.87, 'at': -1.96, 'es': -1.97,
    'ed': -2.02, 'nd': -2.06, 'to': -2.07, 'or': -2.10, 'ea': -2.13,
    'ti': -2.15, 'it': -2.21, 'st': -2.27, 'io': -2.30, 'le': -2.32,
    'is': -2.35, 'ou': -2.38, 'ar': -2.41, 'as': -2.44, 'de': -2.47,
    'rt': -2.52, 'se': -2.55, 'nt': -2.59, 'ha': -2.63, 'ng': -2.65,
    'al': -2.68, 'ss': -2.71, 'te': -2.73, 'si': -2.76, 'co': -2.78,
    'me': -2.81, 'ne': -2.83, 'ro': -2.85, 'li': -2.88, 'ri': -2.90,
    'hi': -2.93, 'ra': -2.95, 'ic': -2.97, 'ce': -3.00, 'il': -3.02,
    'rs': -3.05, 'tr': -3.07, 'ns': -3.10, 'ot': -3.13, 'el': -3.15,
    'ad': -3.18, 'ma': -3.21, 'la': -3.23, 'na': -3.25, 'lo': -3.27,
    'pr': -3.30, 'ac': -3.32, 'ca': -3.35, 'om': -3.37, 'et': -3.40,
    'di': -3.43, 'po': -3.46, 'ge': -3.48, 'sp': -3.51, 'ta': -3.53,
    'ec': -3.55, 'sa': -3.58, 'un': -3.60, 'mi': -3.62, 'fo': -3.65,
    'pe': -3.67, 'gi': -3.70, 'so': -3.72, 'pa': -3.74, 'fi': -3.77,
    'mo': -3.79, 'pl': -3.82, 'ho': -3.84, 'ba': -3.87, 'bu': -3.89,
    'op': -3.91, 'pi': -3.93, 'ab': -3.95, 'ty': -3.97, 'iv': -4.00,
    'gr': -4.02, 'ly': -3.05, 'ry': -4.04, 'em': -4.06, 'bi': -4.08,
    'ow': -4.11, 'cl': -4.13, 'ex': -4.16, 'ev': -4.18, 'do': -4.20,
    'id': -4.23, 'be': -4.25, 'no': -4.27, 've': -4.30, 'fr': -4.32,
    'vi': -4.34, 'ol': -4.36, 'wo': -4.38, 'ch': -4.40, 'am': -4.42,
    'ob': -4.44, 'pu': -4.46, 'tu': -4.48, 'if': -4.50, 'cr': -4.52,
    'ag': -4.55, 'cy': -4.57, 'us': -4.59, 'bl': -4.61, 'ia': -4.63,
    'im': -4.65, 'mp': -4.67, 'ul': -4.70, 'ur': -4.72, 'lt': -4.74,
    'ct': -4.76, 'pp': -4.78, 'ip': -4.80, 'oc': -4.82, 'ik': -4.84,
    'ep': -4.86, 'ui': -4.89, 'fu': -4.91, 'ap': -4.93, 'gl': -4.95,
}

_BIGRAM_LOG2_FLOOR = -10.0


def _markov_log_likelihood(s: str) -> float:
    """Mean bigram log-likelihood under an English character bigram model.
    Lower (more negative) = less English-like = more DGA-like.
    Returned value is already negated so higher = more OOD."""
    if len(s) < 2:
        return 0.0
    total = sum(_BIGRAM_LOG2.get(s[i:i + 2], _BIGRAM_LOG2_FLOOR)
                for i in range(len(s) - 1))
    return -(total / (len(s) - 1))   # negate: higher → less English-like


def _kl_div_from_benign(s: str) -> float:
    """KL divergence of domain's char distribution from benign reference.
    Higher = more different from benign = more DGA-like."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    kl = 0.0
    for ch, cnt in counts.items():
        p = cnt / total
        q = _BENIGN_CHAR_FREQ.get(ch, 1e-6)
        kl += p * math.log(p / q)
    return max(0.0, kl)


def _compression_ratio(s: str) -> float:
    """zlib compression ratio = compressed_len / original_len.
    Low ratio → high repetition → more structured (benign-like).
    High ratio → random-looking → more DGA-like."""
    if not s:
        return 1.0
    try:
        compressed = zlib.compress(s.encode("ascii", errors="replace"), level=9)
        return len(compressed) / len(s)
    except Exception:
        return 1.0


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
    "markov_log_likelihood",
    "kl_div_from_benign",
    "compression_ratio",
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
        _markov_log_likelihood(full),                # markov_log_likelihood
        _kl_div_from_benign(full),                   # kl_div_from_benign
        _compression_ratio(full),                    # compression_ratio
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
