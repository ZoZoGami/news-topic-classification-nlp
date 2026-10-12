"""
models.py — TensorFlow/Keras Model Factories for News Topic Classification

Reusable builders for the model families compared in the benchmark:

    TF-IDF based:
        - Four-layer Dense Neural Network

    Skip-gram Word2Vec based:
        - SimpleRNN
        - GRU
        - LSTM
        - Bidirectional RNN / GRU / LSTM

Best observed result: Skip-gram + GRU → 91.83% accuracy, 0.918 macro-F1.

Author: Fayez Ahmad Protik
Project: News Topic Classification with Deep Learning
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU, LSTM, SimpleRNN


RecurrentType = Literal["SimpleRNN", "GRU", "LSTM"]


# ---------------------------------------------------------------------------
# TF-IDF + Dense Neural Network
# ---------------------------------------------------------------------------

def build_tfidf_dnn(
    input_dim: int,
    num_classes: int = 4,
    hidden_dims: Sequence[int] = (256, 128, 64, 32),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build a feed-forward classifier for dense TF-IDF features.

    Args:
        input_dim: Number of TF-IDF features.
        num_classes: Number of output classes.
        hidden_dims: Width of each hidden layer; default is four layers.
        dropout_rate: Fraction of hidden units randomly dropped during training.
        learning_rate: Adam optimizer learning rate.

    Returns:
        A compiled Keras classification model.
    """
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer width")

    model = Sequential(name="tfidf_dense_classifier")
    model.add(Dense(hidden_dims[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))

    for units in hidden_dims[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation="softmax", name="topic_probabilities"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Skip-gram Word2Vec + recurrent sequence classifier
# ---------------------------------------------------------------------------

def build_sequence_classifier(
    model_type: RecurrentType,
    embedding_matrix: np.ndarray,
    hidden_units: int = 256,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    max_sequence_length: int = 45,
    num_classes: int = 4,
    bidirectional: bool = False,
) -> tf.keras.Model:
    """Build a recurrent headline classifier initialized with Skip-gram weights.

    The embedding layer begins with weights learned by a Word2Vec Skip-gram
    model trained on the training partition. It remains trainable, allowing the
    representation to adapt to the supervised topic-classification objective.

    Args:
        model_type: `SimpleRNN`, `GRU`, or `LSTM`.
        embedding_matrix: Word2Vec embedding table with shape
            `(vocabulary_size, embedding_dimension)`.
        hidden_units: Hidden-state size for the recurrent layer.
        dropout_rate: Dropout applied inside/after the classifier.
        learning_rate: Adam optimizer learning rate.
        max_sequence_length: Input sequence length after padding/truncation.
        num_classes: Number of topic classes.
        bidirectional: Wrap the recurrent layer in a Bidirectional layer.

    Returns:
        A compiled Keras classification model.
    """
    recurrent_layers = {
        "SimpleRNN": SimpleRNN,
        "GRU": GRU,
        "LSTM": LSTM,
    }
    if model_type not in recurrent_layers:
        supported = ", ".join(recurrent_layers)
        raise ValueError(f"Unsupported model_type '{model_type}'. Use one of: {supported}")

    if embedding_matrix.ndim != 2:
        raise ValueError("embedding_matrix must be a 2D array: (vocabulary_size, embedding_dimension)")

    vocab_size, embedding_dim = embedding_matrix.shape
    recurrent_layer = recurrent_layers[model_type](hidden_units, dropout=dropout_rate)

    model = Sequential(name=f"{'bi_' if bidirectional else ''}{model_type.lower()}_classifier")
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=True,
            name="skipgram_embeddings",
        )
    )
    model.add(Bidirectional(recurrent_layer) if bidirectional else recurrent_layer)
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax", name="topic_probabilities"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(
    model_name: str,
    embedding_matrix: np.ndarray | None = None,
    input_dim: int | None = None,
    **kwargs,
) -> tf.keras.Model:
    """Convenience factory for all benchmark neural architectures.

    Args:
        model_name: One of `DNN`, `SimpleRNN`, `GRU`, `LSTM`, `BiRNN`,
            `BiGRU`, or `BiLSTM`.
        embedding_matrix: Required for recurrent sequence models.
        input_dim: Required for `DNN`; TF-IDF feature count.
        **kwargs: Arguments forwarded to the respective builder.

    Returns:
        A compiled TensorFlow/Keras model.
    """
    if model_name == "DNN":
        if input_dim is None:
            raise ValueError("input_dim is required when model_name='DNN'")
        return build_tfidf_dnn(input_dim=input_dim, **kwargs)

    aliases = {
        "SimpleRNN": ("SimpleRNN", False),
        "GRU": ("GRU", False),
        "LSTM": ("LSTM", False),
        "BiRNN": ("SimpleRNN", True),
        "BiGRU": ("GRU", True),
        "BiLSTM": ("LSTM", True),
    }
    if model_name not in aliases:
        valid = ", ".join(["DNN", *aliases])
        raise ValueError(f"Unknown model_name '{model_name}'. Use one of: {valid}")
    if embedding_matrix is None:
        raise ValueError(f"embedding_matrix is required when model_name='{model_name}'")

    model_type, bidirectional = aliases[model_name]
    return build_sequence_classifier(
        model_type=model_type,
        embedding_matrix=embedding_matrix,
        bidirectional=bidirectional,
        **kwargs,
    )
