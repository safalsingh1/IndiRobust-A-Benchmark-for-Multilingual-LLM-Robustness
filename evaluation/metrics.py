from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Dict, Any, Union
import numpy as np

def calculate_classification_metrics(references: List[Any], predictions: List[Any]) -> Dict[str, float]:
    """
    Calculate standard classification metrics: Accuracy and Macro F1.
    
    Definitions:
    - Accuracy: (TP + TN) / (P + N). The proportion of correctly classified samples.
    - Macro F1: Arithmetic mean of F1 scores for each class: (1/C) * sum(F1_c).
      Treats all classes equally regardless of support.
      
    Intuition:
    - Accuracy gives a general sense of correctness but can be misleading for imbalanced datasets.
    - Macro F1 penalizes models that ignore minority classes (e.g., in skewed NLI datasets).
    
    Args:
        references (List[Any]): Ground truth labels (y_true).
        predictions (List[Any]): Model predictions (y_pred).
        
    Returns:
        Dict[str, float]: 
            - 'accuracy': Score in [0, 1].
            - 'f1_macro': Score in [0, 1].
            - 'confusion_matrix': List[List[int]] (C x C matrix).
    """
    # Ensure lists are compatible length
    if len(references) != len(predictions):
        raise ValueError(f"Length mismatch: refs {len(references)} vs preds {len(predictions)}")
    
    if len(references) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    # Handle potential label formats? (String vs Int). SCikit-learn handles mixed types poorly usually.
    # We assume runner/loader normalizes this.
    
    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='macro', zero_division=0)
    
    # Confusion Matrix
    # Labels might be missing if batch is small or not all classes present.
    # We let sklearn handle it, but for JSON output we convert to list.
    cm = confusion_matrix(references, predictions)
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist()
    }

def calculate_consistency(clean_preds: List[Any], perturbed_preds: List[Any]) -> float:
    """
    Calculate Consistency Score.
    
    Definition:
    - Consistency = (1 / N) * Σ I(f(x_i) = f(x_i'))
      where f(x) is the prediction on clean input and f(x') on perturbed input.
      
    Intuition:
    - Measures stability of the model's decision boundary. 
    - A model can be consistent even if wrong (predicts 'Wrong' for both clean and perturbed).
    - High accuracy but low consistency implies the model is lucky or fragile.
    
    Args:
        clean_preds (List[Any]): Predictions on clean dataset.
        perturbed_preds (List[Any]): Predictions on perturbed dataset.
        
    Returns:
        float: Consistency score in [0, 1].
    """
    if len(clean_preds) != len(perturbed_preds):
        return 0.0
    if not clean_preds:
        return 0.0
        
    agreements = sum(1 for c, p in zip(clean_preds, perturbed_preds) if c == p)
    return float(agreements) / len(clean_preds)

def calculate_robustness_summary(clean_metrics: Dict[str, float], perturbed_metrics: Dict[str, float], consistency_score: float = None) -> Dict[str, float]:
    """
    Compute summary of robustness drops (Relative and Absolute).
    
    Definitions:
    - Absolute Drop (Δ_abs): metric_clean - metric_perturbed
    - Relative Drop (Δ_rel): (metric_clean - metric_perturbed) / metric_clean
    
    Intuition:
    - Absolute drop measures raw performance loss.
    - Relative drop normalizes for the baseline performance, allowing comparison across 
      tasks with different difficulty levels (e.g., a 5% drop from 90% is less severe 
      than a 5% drop from 55% in relative terms? Or arguably more? 
      Here, relative drop highlights the *percentage of capability lost*).
      
    Args:
        clean_metrics (Dict): Metrics dict from clean run.
        perturbed_metrics (Dict): Metrics dict from perturbed run.
        consistency_score (float, optional): Pre-calculated consistency.
        
    Returns:
        Dict[str, float]: Summary dictionary with 'abs_drop_*' and 'rel_drop_*'.
    """
    summary = {}
    
    # Accuracy Drop
    clean_acc = clean_metrics.get('accuracy', 0.0)
    pert_acc = perturbed_metrics.get('accuracy', 0.0)
    
    summary['acc_clean'] = clean_acc
    summary['acc_perturbed'] = pert_acc
    summary['abs_drop_acc'] = clean_acc - pert_acc
    summary['rel_drop_acc'] = (clean_acc - pert_acc) / clean_acc if clean_acc > 0 else 0.0
    
    # F1 Drop
    clean_f1 = clean_metrics.get('f1_macro', 0.0)
    pert_f1 = perturbed_metrics.get('f1_macro', 0.0)
    
    summary['f1_clean'] = clean_f1
    summary['f1_perturbed'] = pert_f1
    summary['abs_drop_f1'] = clean_f1 - pert_f1
    summary['rel_drop_f1'] = (clean_f1 - pert_f1) / clean_f1 if clean_f1 > 0 else 0.0
    
    if consistency_score is not None:
        summary['consistency'] = consistency_score
        
    return summary
