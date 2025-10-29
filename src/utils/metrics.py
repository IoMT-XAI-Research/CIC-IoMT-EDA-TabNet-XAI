"""
Metrics calculation utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, Any, List, Optional


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: Optional[np.ndarray] = None,
                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        class_names: Class names for reporting
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Add ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
            metrics['roc_auc_micro'] = roc_auc_score(y_true, y_proba, average='micro', multi_class='ovr')
        except ValueError:
            # Handle case where ROC AUC cannot be calculated
            metrics['roc_auc_macro'] = None
            metrics['roc_auc_micro'] = None
    
    # Add classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    
    # Add confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics









