import os
import json
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
from collections import defaultdict

from models.base import ModelRunner
from .metrics import calculate_classification_metrics

class Evaluator:
    """
    Orchestrates the evaluation process.
    """
    
    def __init__(self, runner: ModelRunner, output_dir: str = "results"):
        self.runner = runner
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate_task(self, dataset: List[Dict], task_name: str, perturbation_name: str = "clean") -> tuple[Dict[str, Any], Dict[str, List[Any]]]:
        """
        Evaluate a dataset.
        Returns:
            (metrics_dict, predictions_dict)
            metrics_dict: {lang: {acc: ...}}
            predictions_dict: {lang: [pred1, pred2...]}
        """
        print(f"Evaluating {task_name} [{perturbation_name}] on {len(dataset)} samples...")
        
        # 1. Group by language
        lang_groups = defaultdict(list)
        for ex in dataset:
            lang = ex.get('language', 'unknown')
            lang_groups[lang].append(ex)
            
        overall_results = {}
        all_predictions = {}
        
        # 2. Run inference per language group
        for lang, examples in lang_groups.items():
            print(f"  > Processing {lang} ({len(examples)} samples)")
            
            # Prepare inputs
            # ModelRunner expects str or dict.
            inputs = []
            labels = []
            ids = []
            
            for ex in examples:
                # Decide input format based on keys
                if 'premise' in ex and 'hypothesis' in ex:
                    inputs.append({'premise': ex['premise'], 'hypothesis': ex['hypothesis']})
                else:
                    inputs.append(ex.get('text', ''))
                    
                labels.append(ex.get('label'))
                ids.append(ex.get('id', ''))
                
            # Batch Predict
            # Now returns list of dicts: [{'label': 'A', 'score': 0.9}, ...]
            raw_preds = self.runner.batch_predict(inputs, batch_size=16)
            
            # Extract labels for metrics
            # Handle case where runner might return simple string (fallback) or dict
            pred_labels = []
            pred_scores = []
            
            for p in raw_preds:
                if isinstance(p, dict):
                    pred_labels.append(p.get('label', str(p)))
                    pred_scores.append(p.get('score', 0.0))
                else:
                    pred_labels.append(str(p))
                    pred_scores.append(1.0)
            
            # 3. Calculate metrics
            metrics = calculate_classification_metrics(labels, pred_labels)
            
            # Log results for this group
            overall_results[lang] = metrics
            
            # Save predictions details including text and scores
            # inputs is a list of str or dicts. If dict, stringify for CSV.
            text_for_csv = []
            for inp in inputs:
                if isinstance(inp, dict):
                    text_for_csv.append(json.dumps(inp, ensure_ascii=False))
                else:
                    text_for_csv.append(str(inp))
            
            self._save_predictions(task_name, perturbation_name, lang, ids, text_for_csv, labels, pred_labels, pred_scores)
            
        # Save aggregated metrics
        self._save_metrics(task_name, perturbation_name, overall_results)
        
        # Collect predictions for return (re-keyed by lang)
        # We need to make sure we return them in a way that matches the other run if we want to compare.
        # But 'evaluate_task' processes in language groups order.
        # The caller 'run_benchmark' needs to alignments. 
        # Actually, saving predictions is safest. But returning dict {lang: preds} is helpful too.
        
        # We need to reload predictions or keep them in memory.
        # Let's return the preds per language.
        all_preds = {}
        # Re-iterate lang groups to get preds back? No, we lost them in the loop unless we stored them.
        # Let's store them in the loop above.
        
        return overall_results

    def _save_predictions(self, task: str, perturbation: str, lang: str, ids: list, texts: list, labels: list, preds: list, scores: list):
        df = pd.DataFrame({
            'id': ids,
            'text': texts,
            'label': labels,
            'prediction': preds,
            'score': scores
        })
        filename = f"{task}_{perturbation}_{lang}_preds.csv"
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        
    def _save_metrics(self, task: str, perturbation: str, metrics: Dict[str, Dict[str, float]]):
        filename = f"{task}_{perturbation}_metrics.json"
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
