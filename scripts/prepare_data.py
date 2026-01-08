import sys
import os
import json
import logging
from collections import Counter
from datasets import DatasetDict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loaders.indicglue import IndicGLUELoader
from data.preprocessing import preprocess_example

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def save_stats(stats: dict):
    stats_path = os.path.join(OUTPUT_DIR, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {stats_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    loader = IndicGLUELoader()
    tasks_to_process = [
        ("wnli", "nli"),
        ("snp", "classification") 
    ]
    
    all_stats = {}

    for task_name, task_type in tasks_to_process:
        logger.info(f"Processing task: {task_name} ({task_type})")
        
        try:
            # Load raw (and standardized schema)
            # NOTE: For real run, ensure 'indic_glue' is installed/accessible.
            # If default load fails (e.g. manual download needed), we catch it.
            dataset = loader.load_task(task_name, task_type)
        except Exception as e:
            logger.error(f"Failed to load {task_name}: {e}")
            continue
            
        # Apply Text Normalization
        dataset = dataset.map(lambda x: preprocess_example(x, task_type))
        
        # Standardize Splits
        # If no validation set, create from train (e.g., 10%)
        if "validation" not in dataset:
            if "train" in dataset:
                logger.info(f"Creating validation split for {task_name}")
                split = dataset["train"].train_test_split(test_size=0.1, seed=42)
                dataset["train"] = split["train"]
                dataset["validation"] = split["test"]
            else:
                logger.warning(f"No train set found for {task_name}, skipping split creation.")

        # Save to JSONL
        task_stats = {}
        for split_name in dataset.keys():
            output_file = os.path.join(OUTPUT_DIR, f"{task_name}_{split_name}.jsonl")
            dataset[split_name].to_json(output_file, orient="records", lines=True, force_ascii=False)
            
            # Collect stats
            count = len(dataset[split_name])
            langs = Counter(dataset[split_name]['language']) if 'language' in dataset[split_name].column_names else "N/A"
            
            task_stats[split_name] = {
                "count": count,
                "languages": dict(langs) if isinstance(langs, Counter) else langs
            }
            logger.info(f"Saved {split_name} to {output_file} (Count: {count})")
            
        all_stats[task_name] = task_stats

    save_stats(all_stats)
    logger.info("Data preprocessing completed.")

if __name__ == "__main__":
    main()
