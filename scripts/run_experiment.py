import sys
import os
import yaml
import argparse
import logging
from itertools import product
import json

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders.indicglue import IndicGLUELoader
from models.hf import HFModelRunner
from scripts.run_benchmark import apply_perturbation
from evaluation.evaluate_classification import evaluate_classification_robustness
from evaluation.evaluate_nli import evaluate_nli_robustness
from evaluation.reporting import save_robustness_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config_path):
    config = load_config(config_path)
    experiment_name = config.get('experiment_name', 'experiment')
    logger.info(f"Starting Experiment: {experiment_name}")
    
    # Base output structure: results/experiment_name
    base_output_dir = config.get('output_dir', os.path.join('results', experiment_name))
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save config snapshot
    with open(os.path.join(base_output_dir, 'config_snapshot.yaml'), 'w') as f:
        yaml.dump(config, f)

    limit = config.get('limit_samples', None)
    if limit is not None and limit <= 0: limit = None
    
    loader = IndicGLUELoader()
    
    # Iterate over Tasks
    for task_cfg in config['tasks']:
        task_name = task_cfg['name']
        task_type = task_cfg.get('type', 'classification')
        
        logger.info(f"=== Loading Task: {task_name} ===")
        try:
            dataset_dict = loader.load_task(task_name, task_type)
            split = "validation" if "validation" in dataset_dict else "train"
            clean_data = [x for x in dataset_dict[split]]
            
            # Filter languages if specified
            if 'languages' in config:
                target_langs = set(config['languages'])
                clean_data = [x for x in clean_data if x.get('language', 'unknown') in target_langs]
                
            if limit:
                clean_data = clean_data[:limit]
                
            logger.info(f"Loaded {len(clean_data)} samples for {task_name}")
            
        except Exception as e:
            logger.error(f"Failed to load task {task_name}: {e}")
            continue

        # Iterate over Models
        for model_cfg in config['models']:
            model_name = model_cfg['name']
            safe_model_name = model_name.replace('/', '_')
            
            logger.info(f"--- Loading Model: {model_name} ---")
            try:
                # Initialize runner
                runner = HFModelRunner(model_name, task="text-classification") 
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue
            
            # Expand perturbations list
            expanded_perts = []
            for p in config['perturbations']:
                p_type = p['type']
                if p_type == 'clean':
                    # We might skip explicit 'clean' perturbation if we rely on the robustness pipeline to run clean vs noisy
                    # But our pipeline takes (clean_data, perturbed_data).
                    # 'clean' perturbation implies clean vs clean? Or just baseline.
                    continue 
                else:
                    levels = p.get('levels', [0.1])
                    for l in levels:
                        expanded_perts.append((p_type, l))
            
            # Run Evaluations per Perturbation Config
            for p_type, level in expanded_perts:
                logger.info(f"Running {p_type} (level={level})...")
                
                # Apply perturbation
                perturbed_data = apply_perturbation(clean_data, p_type, level)
                
                # Select pipeline
                report = {}
                if task_type == 'nli':
                    report = evaluate_nli_robustness(runner, clean_data, perturbed_data, task_name)
                else:
                    report = evaluate_classification_robustness(runner, clean_data, perturbed_data, task_name)
                
                # Save Results
                # Structure: results/experiment/model/
                model_output_dir = os.path.join(base_output_dir, safe_model_name)
                save_robustness_metrics(report, task_name, p_type, level, base_dir=model_output_dir)

    logger.info("Experiment Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to config YAML")
    args = parser.parse_args()
    
    run_experiment(args.config)
