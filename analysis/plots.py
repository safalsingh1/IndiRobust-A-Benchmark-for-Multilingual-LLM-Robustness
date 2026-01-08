import os
import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def parse_metadata(filename):
    """
    Parse task, perturbation, and level from filename.
    Expected format: {task}_{perturbation}_{level}.json
    Example: snp_char_swap_0.1.json
    """
    basename = os.path.basename(filename).replace('.json', '')
    parts = basename.split('_')
    
    # Heuristic parsing
    # Last part is usually level (float)
    try:
        level = float(parts[-1])
        # Join everything else as task_pert
        # But we need to separate task and pert. 
        # Task is usually first token? 'snp', 'wnli', 'xnli'
        # Let's assume known tasks for cleaner parsing, or just heuristic.
        task = parts[0]
        perturbation = "_".join(parts[1:-1])
    except:
        # Fallback for 'clean' or clean_0.0?
        if 'clean' in basename:
            level = 0.0
            perturbation = 'clean'
            task = parts[0]
        else:
            return None, None, None
            
    return task, perturbation, level

def load_data(results_dir):
    files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    data = []
    
    for f in files:
        if "config_snapshot" in f or "robustness_report" in f or "all_results" in f:
            continue
            
        task, pert, level = parse_metadata(f)
        if task is None: continue
        
        try:
            with open(f, 'r') as fd:
                content = json.load(fd)
                
            # Content is {lang: metrics}
            for lang, metrics in content.items():
                if not isinstance(metrics, dict): continue
                
                # Determine parent model folder
                model = os.path.basename(os.path.dirname(f))
                if model == "metrics": model = "unknown"
                
                data.append({
                    "model": model,
                    "task": task,
                    "perturbation": pert,
                    "level": level,
                    "language": lang,
                    "accuracy": metrics.get("acc_perturbed", 0),
                    "rel_drop": metrics.get("rel_drop_acc", 0) * 100,
                    "consistency": metrics.get("consistency", 0) * 100
                })
        except:
            pass
            
    return pd.DataFrame(data)

def plot_performance_vs_noise(df, output_dir):
    """
    Line plot: Accuracy vs Noise Level, grouped by Perturbation/Model.
    """
    if df.empty: return
    
    # Filter out clean (level 0) to avoid skewing unless we treat it as level 0 of all perts
    # Actually, clean is level 0 for all perts.
    
    plt.figure(figsize=(10, 6))
    
    # Plot only if we have levels > 0
    subset = df[df['level'] > 0]
    
    if subset.empty:
        print("Not enough data for Noise Level plot.")
        return

    sns.lineplot(data=subset, x="level", y="accuracy", hue="perturbation", style="model", markers=True, dashes=False)
    
    plt.title("Performance vs Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perf_vs_noise.png"), dpi=300)
    plt.close()

def plot_drop_by_language(df, output_dir):
    """
    Bar plot: Relative drop % per language, averaged across perts.
    """
    if df.empty: return
    
    plt.figure(figsize=(8, 5))
    
    # Exclude clean
    subset = df[df['perturbation'] != 'clean']
    
    if subset.empty: return

    sns.barplot(data=subset, x="language", y="rel_drop", hue="model", errorbar=('ci', 95))
    
    plt.title("Robustness Drop by Language (Lower is Better)")
    plt.xlabel("Language")
    plt.ylabel("Relative Accuracy Drop (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drop_by_language.png"), dpi=300)
    plt.close()

def plot_consistency_vs_type(df, output_dir):
    """
    Bar plot: Consistency Score vs Perturbation Type.
    """
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    subset = df[df['perturbation'] != 'clean']
    
    sns.barplot(data=subset, x="perturbation", y="consistency", hue="language", palette="viridis")
    
    plt.title("Consistency vs Perturbation Type")
    plt.xlabel("Perturbation Type")
    plt.ylabel("Consistency Score (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consistency_vs_type.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_data(args.results_dir)
    print(f"Loaded {len(df)} records.")
    
    plot_performance_vs_noise(df, args.output_dir)
    plot_drop_by_language(df, args.output_dir)
    plot_consistency_vs_type(df, args.output_dir)
    
    print(f"Plots saved to {args.output_dir}")
