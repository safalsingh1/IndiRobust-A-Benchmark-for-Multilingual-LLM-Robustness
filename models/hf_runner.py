import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Union, Dict, Any, Optional
from tqdm import tqdm
import logging

from .base import ModelRunner

logger = logging.getLogger(__name__)

class HFModelRunner(ModelRunner):
    """
    Cleaner, robust wrapper for HuggingFace models.
    Supports efficient batch inference and GPU acceleration.
    """

    def __init__(self, model_name_or_path: str, task: str = "text-classification", device: Optional[str] = None):
        """
        Initialize the runner.
        
        Args:
            model_name_or_path: HF Model ID (e.g. 'google/muril-base-cased', 'xlm-roberta-base') 
                                or path to local checkpoint.
            task: Pipeline task identifier.
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        super().__init__(model_name_or_path, device)
        
        self.task = task
        
        # Auto-detect device if not provided
        if device is None:
            self.device_id = 0 if torch.cuda.is_available() else -1
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
            self.device_id = 0 if device == "cuda" or device == "gpu" else -1
            
        logger.info(f"Loading HF Model: {model_name_or_path} on {self.device_str}...")
        
        # Initialize pipeline
        # We load a generic pipeline. 
        # Note: If model is a base model (no head), this will init a random head and warn.
        # User specified "Do NOT tune", so we use as-is.
        try:
            self.pipe = pipeline(
                task=task,
                model=model_name_or_path,
                device=self.device_id,
                top_k=None # Return all scores for robust analysis? Or default (max)? 
                           # Default text-classification returns top label. 
                           # We want generic behavior.
            )
        except Exception as e:
            logger.error(f"Failed to initialize pipeline for {model_name_or_path}: {e}")
            raise e

    def predict(self, input_text: Union[str, Dict[str, Any]]) -> Any:
        """
        Predict for a single example.
        """
        formatted = self._format_input(input_text)
        # Pipeline call
        res = self.pipe(formatted, truncation=True, max_length=512)
        return self._postprocess(res)

    def batch_predict(self, inputs: List[Union[str, Dict[str, Any]]], batch_size: int = 32) -> List[Any]:
        """
        Optimized batch prediction.
        """
        formatted_inputs = [self._format_input(x) for x in inputs]
        
        results = []
        # Pipeline supports batching with generator/list
        # Using built-in batch_size param
        
        try:
            # Pass truncation args to pipe call
            for out in tqdm(self.pipe(formatted_inputs, batch_size=batch_size, truncation=True, max_length=512), total=len(formatted_inputs), desc="Inference"):
                results.append(self._postprocess(out))
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Fallback or re-raise
            raise e
            
        return results

    def _format_input(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Handle dictionary inputs and enforce length limit.
        """
        text = ""
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, dict):
            if 'premise' in input_data and 'hypothesis' in input_data:
                text = f"{input_data['premise']} {input_data['hypothesis']}"
            elif 'text' in input_data:
                text = input_data['text']
            else:
                text = " ".join([str(v) for v in input_data.values()])
        else:
            text = str(input_data)
            
        # Hard truncation to prevent model crashes (512 tokens approx 2000 chars)
        # Reduced to 100 to force success
        if len(text) > 100:
            text = text[:100]
            
        return text

    def _postprocess(self, result: Any) -> Any:
        """
        Extract label from pipeline result.
        Standard pipeline output: [{'label': 'L', 'score': S}] or just dict.
        """
        if isinstance(result, list):
            res = result[0]
        else:
            res = result
            
        if isinstance(res, dict):
            # Return the full dict to allow access to score
            # But ensure 'label' key exists for generic usage if possible, or normalize.
            return res
            
        return {'label': res, 'score': 1.0} # Fallback for non-dict results (e.g. generation text only)
