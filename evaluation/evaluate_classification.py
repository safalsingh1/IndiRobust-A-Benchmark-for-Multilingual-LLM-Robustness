from typing import List, Dict, Any
from .evaluator import Evaluator
from .metrics import calculate_classification_metrics, calculate_robustness_summary, calculate_consistency
from models.base import ModelRunner

def evaluate_classification_robustness(
    runner: ModelRunner,
    clean_dataset: List[Dict[str, Any]],
    perturbed_dataset: List[Dict[str, Any]],
    task_name: str = "classification"
) -> Dict[str, Any]:
    """
    Evaluate robustness for text classification.
    
    Args:
        runner: ModelRunner instance.
        clean_dataset: List of clean examples.
        perturbed_dataset: List of perturbed examples (paired 1-to-1 with clean).
        task_name: Label for the task.
        
    Returns:
        Dictionary containing robustness report per language.
    """
    evaluator = Evaluator(runner)
    
    # 1. Run Clean Evaluation
    print(f"[{task_name}] Running Clean Evaluation...")
    clean_metrics, clean_preds = evaluator.evaluate_task(clean_dataset, task_name, "clean")
    
    # 2. Run Perturbed Evaluation
    print(f"[{task_name}] Running Perturbed Evaluation...")
    pert_metrics, pert_preds = evaluator.evaluate_task(perturbed_dataset, task_name, "perturbed")
    
    # 3. Compute Robustness Metrics
    report = {}
    all_langs = set(clean_metrics.keys()) | set(pert_metrics.keys())
    
    for lang in all_langs:
        c_met = clean_metrics.get(lang, {})
        p_met = pert_metrics.get(lang, {})
        c_p = clean_preds.get(lang, [])
        p_p = pert_preds.get(lang, [])
        
        # Consistency
        consistency = calculate_consistency(c_p, p_p)
        
        # Summary (Drops)
        summary = calculate_robustness_summary(c_met, p_met, consistency)
        report[lang] = summary
        
    return report
