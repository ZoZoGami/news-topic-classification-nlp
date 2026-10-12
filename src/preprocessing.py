"""
preprocessing.py — Text Preprocessing Pipeline for News Topic Classification

Implements three preprocessing strategies to investigate how text normalization
affects classification performance on short news headlines:

    1. No Preprocessing (raw text passed directly)
    2. Extreme Preprocessing (aggressive cleaning + stemming + lemmatization)
    3. Optimum Preprocessing (targeted HTML/artifact removal, preserves semantic context)

Author: Fayez Ahmad Protik
Project: News Topic Classification with Deep Learning
"""

import re
import html
from typing import Callable, Dict

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    _stemmer = PorterStemmer()
    _lemmatizer = WordNetLemmatizer()
except ImportError:
    _stemmer = None
    _lemmatizer = None


# ---------------------------------------------------------------------------
# Core text cleaning utilities
# ---------------------------------------------------------------------------

def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities from raw text."""
    text = html.unescape(str(text))
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    return text


def remove_wrapper_text(text: str) -> str:
    """Strip boilerplate wrapper strings injected by the data source."""
    text = re.sub(r"News Headlines:", " ", text, flags=re.IGNORECASE)
    return text


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs from text."""
    return re.sub(r"https?://\S+", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Lemmatization / Stemming helpers
# ---------------------------------------------------------------------------

def safe_lemmatize(token: str) -> str:
    """Apply WordNet lemmatization with graceful fallback."""
    if _lemmatizer is None:
        return token
    try:
        return _lemmatizer.lemmatize(token, pos="v")
    except Exception:
        return token


def safe_stem(token: str) -> str:
    """Apply Porter stemming with graceful fallback."""
    if _stemmer is None:
        return token
    return _stemmer.stem(token)


# ---------------------------------------------------------------------------
# Three preprocessing strategies
# ---------------------------------------------------------------------------

def no_preprocess(text: str) -> str:
    """Strategy 1: Return text as-is for baseline comparison.

    This intentionally preserves HTML artifacts, casing, and all noise.
    Useful for establishing whether learned embeddings can handle raw input.
    """
    return str(text)


def extreme_preprocess(text: str) -> str:
    """Strategy 2: Aggressive text normalization.

    Pipeline:
        HTML removal → lowercase → URL removal → punctuation removal →
        number removal → stopword removal → lemmatization → stemming

    This tests whether over-aggressive cleaning helps or hurts.
    """
    text = strip_html(text)
    text = remove_wrapper_text(text)
    text = text.lower()
    text = remove_urls(text)
    text = re.sub(r"[^a-z\s]", " ", text)                    # strip punctuation & digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    tokens = [safe_lemmatize(t) for t in tokens]
    tokens = [safe_stem(t) for t in tokens]
    return normalize_whitespace(" ".join(tokens))


def optimum_preprocess(text: str) -> str:
    """Strategy 3: Balanced cleaning informed by EDA findings.

    Pipeline:
        HTML & wrapper removal → lowercase → URL removal → symbol cleanup →
        preserve numbers, apostrophes, and named entities

    Rationale:
        - HTML tags and 'News Headlines:' wrapper are noise (EDA finding)
        - Numbers carry topic signal (e.g., financial figures for Business)
        - Stopword removal and stemming are skipped to preserve entities
    """
    text = strip_html(text)
    text = remove_wrapper_text(text)
    text = text.lower()
    text = remove_urls(text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)                # keep digits & apostrophes
    return normalize_whitespace(text)


# ---------------------------------------------------------------------------
# Registry for easy iteration during experiments
# ---------------------------------------------------------------------------

PREPROCESSORS: Dict[str, Callable[[str], str]] = {
    "raw":     no_preprocess,
    "extreme": extreme_preprocess,
    "optimum": optimum_preprocess,
}


def apply_preprocessing(texts: list, strategy: str = "optimum") -> list:
    """Apply a named preprocessing strategy to a list of texts.

    Args:
        texts: List of raw text strings.
        strategy: One of 'raw', 'extreme', or 'optimum'.

    Returns:
        List of preprocessed text strings.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    if strategy not in PREPROCESSORS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(PREPROCESSORS.keys())}"
        )
    func = PREPROCESSORS[strategy]
    return [func(t) for t in texts]
