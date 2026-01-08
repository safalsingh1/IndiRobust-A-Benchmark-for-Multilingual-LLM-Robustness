from typing import List, Dict, Any
from .evaluator import Evaluator
from .metrics import calculate_classification_metrics, calculate_robustness_summary, calculate_consistency
from models.base import ModelRunner

def evaluate_nli_robustness(
    runner: ModelRunner,
    clean_dataset: List[Dict[str, Any]],
    perturbed_dataset: List[Dict[str, Any]],
    task_name: str = "nli"
) -> Dict[str, Any]:
    """
    Evaluate robustness for NLI.
    
    Args:
        runner: ModelRunner instance.
        clean_dataset: List of clean examples (must have 'premise', 'hypothesis' or mapped 'text').
        perturbed_dataset: List of perturbed examples.
        task_name: Label for the task.
        
    Returns:
        Dictionary containing robustness report per language.
    """
    # NLI is technically classification (entailment/neutral/contradiction).
    # We use the same base evaluator which handles input formatting.
    # We separate it to allow future NLI-specific metrics (like entailment-specific consistency)
    # or different confusion matrix handling.
    
    evaluator = Evaluator(runner)
    
    # 1. Run Clean
    print(f"[{task_name}] Running Clean NLI Evaluation...")
    clean_metrics, clean_preds = evaluator.evaluate_task(clean_dataset, task_name, "clean")
    
    # 2. Run Perturbed
    print(f"[{task_name}] Running Perturbed NLI Evaluation...")
    pert_metrics, pert_preds = evaluator.evaluate_task(perturbed_dataset, task_name, "perturbed")
    
    # 3. Robustness
    report = {}
    all_langs = set(clean_metrics.keys()) | set(pert_metrics.keys())
    
    for lang in all_langs:
        c_met = clean_metrics.get(lang, {})
        p_met = pert_metrics.get(lang, {})
        c_p = clean_preds.get(lang, [])
        p_p = pert_preds.get(lang, [])
        
        consistency = calculate_consistency(c_p, p_p)
        summary = calculate_robustness_summary(c_met, p_met, consistency)
        report[lang] = summary
        
    return report
