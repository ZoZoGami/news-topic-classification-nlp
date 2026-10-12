"""
evaluation.py — Evaluation Utilities for News Topic Classification

Provides standardized evaluation, visualization, and result logging functions
used across all 24 experimental combinations in the project.

Metrics:
    - Accuracy
    - Macro F1-Score (primary metric — accounts for class imbalance)
    - Per-class Precision / Recall / F1
    - Confusion Matrix visualization

Author: Fayez Ahmad Protik
Project: News Topic Classification with Deep Learning
"""

import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Default class names for the news topic dataset
# ---------------------------------------------------------------------------

DEFAULT_CLASS_NAMES = [
    "Business",
    "Science and Technology",
    "Sports",
    "World News",
]


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    title: str = "Model Evaluation",
    verbose: bool = True,
) -> Dict:
    """Compute and optionally display classification metrics.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        class_names: Human-readable class names for the report.
        title:       Title printed above metrics output.
        verbose:     Whether to print the report to stdout.

    Returns:
        Dictionary containing 'accuracy', 'macro_f1', 'report' (dict), and
        'confusion_matrix' (np.ndarray).
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Macro F1:   {f1:.4f}")
        print(f"{'=' * 60}")
        print(classification_report(y_true, y_pred, target_names=class_names))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Confusion matrix visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6),
    cmap: str = "Blues",
) -> None:
    """Plot a labeled confusion matrix heatmap.

    Args:
        cm:          Confusion matrix as a 2D numpy array.
        class_names: Labels for each axis.
        title:       Plot title.
        save_path:   If provided, save the figure to this path.
        figsize:     Matplotlib figure size.
        cmap:        Colormap for the heatmap.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=14, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells with count values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Experiment result logger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """Accumulates results from multiple experiments for comparison.

    Usage:
        logger = ExperimentLogger()
        logger.log("raw", "TF-IDF", "MultinomialNB", {...}, 0.90, 0.90, 0.90, 0.90)
        ...
        results_df = logger.to_dataframe()
        logger.plot_comparison()
    """

    def __init__(self):
        self._results: List[Dict] = []

    def log(
        self,
        preprocessing: str,
        representation: str,
        model_name: str,
        best_params: Dict,
        val_acc: float,
        val_f1: float,
        test_acc: float,
        test_f1: float,
        test_report: Optional[Dict] = None,
        test_cm: Optional[np.ndarray] = None,
    ) -> None:
        """Record a single experiment's results.

        Args:
            preprocessing:   Preprocessing strategy name.
            representation:  Feature representation technique.
            model_name:      Model architecture name.
            best_params:     Best hyperparameters from tuning.
            val_acc/val_f1:  Validation set scores.
            test_acc/test_f1: Held-out test set scores.
            test_report:     Full classification report dict (optional).
            test_cm:         Confusion matrix array (optional).
        """
        self._results.append({
            "preprocessing": preprocessing,
            "representation": representation,
            "model": model_name,
            "best_params": best_params,
            "val_accuracy": round(val_acc, 4),
            "val_f1_macro": round(val_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "test_f1_macro": round(test_f1, 4),
            "test_report": test_report,
            "test_confusion_matrix": test_cm,
        })
        print(
            f"  Logged: {preprocessing:12s} | {representation:10s} | "
            f"{model_name:12s} → Test F1: {test_f1:.4f}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert logged results to a pandas DataFrame, sorted by test F1."""
        df = pd.DataFrame(self._results)
        display_cols = [
            "preprocessing", "representation", "model",
            "val_accuracy", "val_f1_macro",
            "test_accuracy", "test_f1_macro",
        ]
        return df[display_cols].sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    def plot_comparison(
        self,
        metric: str = "test_f1_macro",
        title: str = "Model Performance Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """Generate a grouped bar chart comparing all experiments.

        Args:
            metric:    Column name to compare ('test_f1_macro' or 'test_accuracy').
            title:     Plot title.
            save_path: If provided, save the figure.
        """
        df = self.to_dataframe()
        df["label"] = df["preprocessing"] + "\n" + df["model"]

        fig, ax = plt.subplots(figsize=(16, 6))
        colors = {"TF-IDF": "#4C72B0", "Skip-gram": "#DD8452"}
        for _, row in df.iterrows():
            color = colors.get(row["representation"], "#666666")
            ax.bar(row["label"], row[metric], color=color, edgecolor="white")

        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, output_dir: str = ".") -> None:
        """Save results table and reports to disk.

        Args:
            output_dir: Directory for output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        df = self.to_dataframe()
        csv_path = os.path.join(output_dir, "experiment_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Results table saved: {csv_path}")

        reports = {
            f"{r['preprocessing']}_{r['representation']}_{r['model']}": r.get("test_report")
            for r in self._results
            if r.get("test_report") is not None
        }
        if reports:
            reports_path = os.path.join(output_dir, "classification_reports.json")
            with open(reports_path, "w") as f:
                json.dump(reports, f, indent=2)
            print(f"  Reports saved: {reports_path}")
