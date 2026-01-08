import pandas as pd
import json
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ConsistencyAnalyzer:
    """
    Analyzes prediction consistency between clean and perturbed runs.
    """
    def __init__(self):
        pass

    def analyze_pair(self, clean_preds_path: str, pert_preds_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Compare two prediction CSVs (clean vs perturbed).
        Expects CSVs to have 'id', 'prediction', 'label' columns.
        
        Args:
            clean_preds_path: Path to clean predictions CSV.
            pert_preds_path: Path to perturbed predictions CSV.
            output_path: Optional path to save detailed report.
            
        Returns:
            Dictionary with aggregate stats (consistency, flip_rate).
        """
        try:
            clean_df = pd.read_csv(clean_preds_path)
            pert_df = pd.read_csv(pert_preds_path)
            
            # Ensure IDs are strings and indices for alignment
            clean_df['id'] = clean_df['id'].astype(str)
            pert_df['id'] = pert_df['id'].astype(str)
            
            # Merge on ID
            # robust merge handling potential missing rows in one or the other
            merged = pd.merge(clean_df, pert_df, on='id', suffixes=('_clean', '_pert'), how='inner')
            
            if len(merged) == 0:
                logger.warning(f"No overlapping IDs found between {clean_preds_path} and {pert_preds_path}")
                return {"consistency": 0.0, "flip_rate": 0.0, "total_samples": 0}
            
            # Calculate Consistency: Prediction(clean) == Prediction(pert)
            merged['is_consistent'] = merged['prediction_clean'] == merged['prediction_pert']
            
            total = len(merged)
            consistent_count = merged['is_consistent'].sum()
            consistency = consistent_count / total
            flip_rate = 1.0 - consistency
            
            stats = {
                "total_samples": total,
                "consistent_count": int(consistent_count),
                "flipped_count": int(total - consistent_count),
                "consistency_score": float(consistency),
                "flip_rate": float(flip_rate)
            }
            
            if output_path:
                self._save_detailed_report(merged, stats, output_path)
                
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing consistency: {e}")
            return {}

    def _save_detailed_report(self, merged_df: pd.DataFrame, stats: Dict, output_path: str):
        """
        Save detailed Per-Example analysis.
        """
        # prepare list of dicts
        details = []
        for _, row in merged_df.iterrows():
            details.append({
                "id": row['id'],
                "label": row['label_clean'], # Assuming label consistent
                "clean_prediction": row['prediction_clean'],
                "perturbed_prediction": row['prediction_pert'],
                "is_consistent": bool(row['is_consistent'])
            })
            
        report = {
            "summary": stats,
            "details": details
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Detailed consistency report saved to {output_path}")

def run_analysis_on_directory(results_dir: str, analysis_output_dir: str):
    """
    Batch process all corresponding prediction files in results directory.
    """
    import glob
    
    analyzer = ConsistencyAnalyzer()
    
    # Simple heuristic: find all clean files, then look for their perturbed counterparts
    # File pattern from Evaluator: {task}_{perturbation}_{lang}_preds.csv
    
    clean_files = glob.glob(os.path.join(results_dir, "*_clean_*_preds.csv"))
    
    for clean_path in clean_files:
        basename = os.path.basename(clean_path)
        # Parse: {task}_clean_{lang}_preds.csv
        try:
            parts = basename.split('_clean_')
            task = parts[0]
            lang = parts[1].replace('_preds.csv', '')
            
            # Construct matching pattern for this task/lang but with any perturbation
            # glob pattern: {task}_*_{lang}_preds.csv
            # But we want to exclude 'clean'
            
            all_task_lang_files = glob.glob(os.path.join(results_dir, f"{task}_*_{lang}_preds.csv"))
            
            for pert_path in all_task_lang_files:
                if pert_path == clean_path:
                    continue
                    
                # Extract perturbation name
                # pert filename: {task}_{pert_name}_{lang}_preds.csv
                # This parsing is tricky if pert_name has underscores.
                # Regex might be safer, or just splitting.
                # let's assume str replace logic
                pert_filename = os.path.basename(pert_path)
                pert_name = pert_filename.replace(f"{task}_", "").replace(f"_{lang}_preds.csv", "")
                
                output_filename = f"consistency_{task}_{pert_name}_{lang}.json"
                output_path = os.path.join(analysis_output_dir, output_filename)
                
                print(f"Analyzing consistency: {task} | {lang} | Clean vs {pert_name}")
                analyzer.analyze_pair(clean_path, pert_path, output_path)
                
        except Exception as e:
            logger.error(f"Failed to parse or process {basename}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="analysis/consistency_reports")
    args = parser.parse_args()
    
    run_analysis_on_directory(args.results_dir, args.output_dir)
