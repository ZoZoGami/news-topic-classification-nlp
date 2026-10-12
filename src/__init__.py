"""
News Topic Classification with Deep Learning.

An end-to-end NLP benchmark comparing TF-IDF and Skip-gram Word2Vec
representations across Naive Bayes, dense neural networks, RNNs, GRUs,
LSTMs, and bidirectional recurrent variants for multiclass headline
classification.

Best observed result: GRU + Skip-gram (raw text) → 91.83% accuracy,
0.918 macro-F1.
"""

from .preprocessing import apply_preprocessing, PREPROCESSORS
from .models import build_model, build_tfidf_dnn, build_sequence_classifier
from .evaluation import evaluate_predictions, plot_confusion_matrix, ExperimentLogger

__version__ = "1.0.0"

__all__ = [
    "apply_preprocessing",
    "PREPROCESSORS",
    "build_model",
    "build_tfidf_dnn",
    "build_sequence_classifier",
    "evaluate_predictions",
    "plot_confusion_matrix",
    "ExperimentLogger",
]
