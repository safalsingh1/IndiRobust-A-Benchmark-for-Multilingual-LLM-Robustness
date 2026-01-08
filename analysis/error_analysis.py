import os
import pandas as pd
import json
import argparse
import glob
from collections import defaultdict

def run_deep_error_analysis(results_dir: str, output_file: str, high_conf_threshold: float = 0.9):
    """
    Analyze predictions for Flips and High Confidence Failures.
    Requires CSVs with 'text' and 'score' columns.
    """
    print(f"Scanning {results_dir} ...")
    
    files = glob.glob(os.path.join(results_dir, "**", "*_preds.csv"), recursive=True)
    
    # Map: task -> lang -> perturbation -> file
    file_map = defaultdict(lambda: defaultdict(dict))
    
    for f in files:
        basename = os.path.basename(f)
        # Assuming format {task}_{perturbation}_{lang}_preds.csv
        # If deeply nested, might be hard to parse perturbation cleanly if underscores exist
        # But usually 'clean' is distinct.
        
        try:
            # We look for "clean" or others.
            # Let's try to extract lang (last part)
            parts = basename.replace('_preds.csv', '').split('_')
            lang = parts[-1]
            task = parts[0]
            pert = "_".join(parts[1:-1])
            
            file_map[task][lang][pert] = f
        except:
            continue
            
    error_cases = []

    for task, lang_dict in file_map.items():
        for lang, pert_dict in lang_dict.items():
            if 'clean' not in pert_dict: continue
            
            try:
                clean_df = pd.read_csv(pert_dict['clean'])
                # Ensure str IDs
                clean_df['id'] = clean_df['id'].astype(str)
                
                for pert_name, pert_file in pert_dict.items():
                    if pert_name == 'clean': continue
                    
                    pert_df = pd.read_csv(pert_file)
                    pert_df['id'] = pert_df['id'].astype(str)
                    
                    # Merge on ID to compare same samples
                    # We want text from both if available? 
                    # Usually text is in both if Evaluator saved it.
                    # text_clean vs text_pert
                    
                    merged = pd.merge(clean_df, pert_df, on='id', suffixes=('_clean', '_pert'))
                    
                    # 1. Flips
                    flips = merged[merged['prediction_clean'] != merged['prediction_pert']]
                    
                    # 2. High Conf Failures
                    # Where Perturbed Label != True Label AND Score > Threshold
                    # Ensure we have label and score columns
                    if 'label_pert' in merged.columns and 'score_pert' in merged.columns:
                        high_conf = merged[
                            (merged['prediction_pert'] != merged['label_pert']) & 
                            (merged['score_pert'] > high_conf_threshold)
                        ]
                    else:
                        high_conf = pd.DataFrame()

                    # Collect Qualitative Examples
                    # Limit to 5 of each type to save space
                    
                    # Process Flips
                    for _, row in flips.head(5).iterrows():
                        error_cases.append({
                            "type": "prediction_flip",
                            "task": task,
                            "language": lang,
                            "perturbation": pert_name,
                            "id": row['id'],
                            "text_clean": row.get('text_clean', ''),
                            "text_perturbed": row.get('text_perturbed', ''),
                            "pred_clean": row['prediction_clean'],
                            "pred_perturbed": row['prediction_pert'],
                            "ground_truth": row.get('label_clean')
                        })
                        
                    # Process High Conf
                    for _, row in high_conf.head(5).iterrows():
                        error_cases.append({
                            "type": "high_confidence_failure",
                            "task": task,
                            "language": lang,
                            "perturbation": pert_name,
                            "id": row['id'],
                            "text_perturbed": row.get('text_perturbed', ''),
                            "prediction": row['prediction_pert'],
                            "confidence": row['score_pert'],
                            "ground_truth": row['label_pert']
                        })
            
            except Exception as e:
                print(f"Error processing {task}-{lang}: {e}")

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(error_cases)} error cases to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="analysis/error_cases.json")
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    
    run_deep_error_analysis(args.results_dir, args.output, args.threshold)
