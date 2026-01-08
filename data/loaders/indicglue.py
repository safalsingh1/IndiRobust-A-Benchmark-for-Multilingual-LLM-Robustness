import datasets
from datasets import load_dataset, DatasetDict
from typing import Dict, List, Optional, Union

class IndicGLUELoader:
    """
    Loader for IndicGLUE datasets with standardized formatting.
    Supports Text Classification and NLI tasks.
    """
    
    # Task mapping to HF config names if needed
    TASK_CONFIGS = {
        "classification": ["snp"], # Sentiment, News Classification etc. Adjust as per specific IndicGLUE subset availability
        "nli": ["wnli", "rte", "xnli"]
    }
    
    def __init__(self, languages: List[str] = ["en", "hi", "mr", "bn"]):
        self.target_languages = languages

    def load_task(self, task_name: str, task_type: str) -> DatasetDict:
        """
        Load a specific task and standardize it.
        
        Args:
            task_name: The specific subset name (e.g., 'wnli', 'snp').
            task_type: 'classification' or 'nli'.
            
        Returns:
            Standardized DatasetDict.
        """
        # Load the raw dataset
        # Note: IndicGLUE usually requires specifying the subset.
        try:
            raw_dataset = load_dataset("indic_glue", task_name)
        except Exception as e:
            print(f"Error loading {task_name}: {e}")
            raise e

        # Filter languages if the dataset separates them or has a language column
        # Many IndicGLUE subsets have a 'language' column or are split by config.
        # Here we assume standard IndicGLUE structure (often has all langs mixed or split).
        
        standardized_dataset = raw_dataset.map(
            lambda x: self._standardize_example(x, task_name, task_type),
            remove_columns=raw_dataset["train"].column_names
        )
        
        # Filter by language if 'language' column exists and is populated
        # This implementation assumes the standardize function populates 'language' correctly
        # We process the filter after mapping to ensure schema consistency
        
        return standardized_dataset

    def _standardize_example(self, example: Dict, task_name: str, task_type: str) -> Dict:
        """
        Convert raw example to standardized format.
        Schema: {id, language, task, text/premise/hypothesis, label}
        """
        std_example = {
            "task": task_name,
            "label": example.get("label", -1),
            "id": str(example.get("idx", example.get("id", -1)))  # Fallback ID
        }

        # Attempt to detect language if not explicit (often in IndicGLUE it's not in the row)
        # For now, we pass 'unknown' if not present, and rely on caller to know which config loaded which lang
        # OR if 'language' is in the dataset (common in polyglot datasets)
        std_example["language"] = example.get("language", "unknown")

        if task_type == "classification":
            # Common keys: text, sentence
            # bbca.hi has 'text'
            std_example["text"] = example.get("text", example.get("sentence", ""))
            
        elif task_type == "nli":
            # Common keys: premise, hypothesis, question, sentence1, sentence2
            # COPA: premise, choice1, choice2, question, label (0/1)
            # We map COPA to premise/hypothesis style for NLI pipeline?
            # COPA is choice. standard NLI is entailment.
            # If we force COPA into NLI pipeline, we need 'premise' and 'hypothesis'.
            # Let's map premise -> premise, choice1/2 -> hypothesis (but we have 2?)
            # Benchmarks usually pick the correct choice as hypothesis?
            # Simplification for robustness check: hypothesis = choice1 + " " + choice2?
            # Or just ignore COPA and use WSTP?
            # Let's try basic mapping.
            
            p = example.get("premise", example.get("sentence1", ""))
            h = example.get("hypothesis", example.get("sentence2", ""))
            
            # WSTP / WNLI: sentence1, sentence2
            if not p and not h:
                # Try WSTP style
                p = example.get("sentence1", "")
                h = example.get("sentence2", "")
                
            std_example["premise"] = p
            std_example["hypothesis"] = h
            
        return std_example

if __name__ == "__main__":
    # Simple test stub
    print("IndicGLUE Loader defined.")
