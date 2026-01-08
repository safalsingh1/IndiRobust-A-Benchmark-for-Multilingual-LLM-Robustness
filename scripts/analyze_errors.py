import os
import pandas as pd
import json
import argparse
import glob
from collections import defaultdict

def analyze_errors(results_dir: str, output_file: str, sample_size: int = 5):
    """
    Analyze prediction flips between clean and perturbed runs.
    """
    print(f"Scanning {results_dir} for result files...")
    
    # Pattern: {task}_{perturbation}_{lang}_preds.csv
    # We need to pair them up.
    # Group by (task, lang)
    
    files = glob.glob(os.path.join(results_dir, "*_preds.csv"))
    
    # Structure: task -> lang -> perturbation -> filepath
    file_map = defaultdict(lambda: defaultdict(dict))
    
    for f in files:
        basename = os.path.basename(f)
        # simplistic parsing assuming no extra underscores in task names suitable for this benchmark
        # format: task_perturbation_lang_preds.csv
        # This might be brittle if perturbation has underscores (e.g. char_swap).
        # Strategy: Valid perturbations are known or 'clean'.
        # Let's split by key keywords.
        
        parts = basename.rsplit('_preds.csv', 1)[0].split('_')
        # parts: [task, ..., lang]
        # lang is last.
        lang = parts[-1]
        task = parts[0]
        perturbation = "_".join(parts[1:-1])
        
        file_map[task][lang][perturbation] = f

    analysis_report = {}

    for task, lang_dict in file_map.items():
        analysis_report[task] = {}
        
        for lang, pert_dict in lang_dict.items():
            if 'clean' not in pert_dict:
                continue
                
            clean_df = pd.read_csv(pert_dict['clean'])
            # Ensure ID is string for merging
            clean_df['id'] = clean_df['id'].astype(str)
            
            lang_report = {}
            
            for pert_name, pert_file in pert_dict.items():
                if pert_name == 'clean': continue
                
                print(f"Analyzing {task} - {lang} - {pert_name}...")
                pert_df = pd.read_csv(pert_file)
                pert_df['id'] = pert_df['id'].astype(str)
                
                # Merge
                merged = pd.merge(clean_df, pert_df, on='id', suffixes=('_clean', '_pert'))
                
                # Identify Flips (Consistency Failures): Clean Pred != Pert Pred
                # We perform this regardless of Ground Truth correctness for "Robustness" analysis
                # But typically we care about Correct -> Incorrect, or just any Change.
                # User asked for "prediction flips".
                
                flips = merged[merged['prediction_clean'] != merged['prediction_pert']]
                
                flip_rate = len(flips) / len(merged) if len(merged) > 0 else 0
                
                # Sample
                samples = []
                if len(flips) > 0:
                    sample_df = flips.sample(min(len(flips), sample_size), random_state=42)
                    for _, row in sample_df.iterrows():
                        samples.append({
                            "id": row['id'],
                            "label": row['label_clean'], # Assume labels same
                            "clean_pred": row['prediction_clean'],
                            "pert_pred": row['prediction_pert']
                        })
                
                lang_report[pert_name] = {
                    "total_samples": len(merged),
                    "flip_count": len(flips),
                    "flip_rate": flip_rate,
                    "samples": samples
                }
            
            if lang_report:
                analysis_report[task][lang] = lang_report

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(analysis_report, f, indent=2)
    
    print(f"Error analysis saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="analysis/error_cases.json")
    args = parser.parse_args()
    
    analyze_errors(args.results_dir, args.output)
