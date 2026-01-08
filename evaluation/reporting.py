import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def save_robustness_metrics(metrics_report: Dict[str, Any], task: str, perturbation: str, noise_level: float, base_dir: str = "results/metrics"):
    """
    Save robustness metrics to a structured JSON file.
    Output format: base_dir/{task}_{perturbation}_{level}.json
    """
    os.makedirs(base_dir, exist_ok=True)
    
    filename = f"{task}_{perturbation}_{noise_level}.json"
    output_path = os.path.join(base_dir, filename)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics_report, f, indent=2)
        logger.info(f"Robustness metrics saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {output_path}: {e}")
