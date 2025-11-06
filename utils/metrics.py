# utils/metrics.py
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(y_true, y_pred, labels=None):
    """
    Generates and returns a classification report.
    y_true: true labels (numpy array)
    y_pred: predicted labels (numpy array)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Simple conversion for a binary classification context
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
        
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return report
