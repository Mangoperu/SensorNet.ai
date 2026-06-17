"""
Model training and evaluation helpers for the TEP fault diagnosis project.
"""

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_model(model, X_test, y_test, label: str = "Model") -> dict:
    """Run predictions and print a standard evaluation report.

    Returns a dict of {accuracy, macro_f1} for easy comparison across models.
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"--- {label} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds))

    return {"accuracy": acc, "macro_f1": macro_f1}


def train_and_save(model, X_train, y_train, save_path: str):
    """Fit a model and persist it to disk with joblib."""
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Model trained and saved to {save_path}")
    return model


def load_model(path: str):
    """Load a previously saved model."""
    return joblib.load(path)
