import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from typing import List, Union, Dict, Any
from tqdm import tqdm
import numpy as np

from .base import ModelRunner

class HFModelRunner(ModelRunner):
    """
    Concrete implementation of ModelRunner for HuggingFace models.
    Supports: classification, text-generation, text2text-generation.
    """

    def __init__(self, model_name_or_path: str, task: str = "text-classification", device: str = "cpu", torch_dtype=None):
        super().__init__(model_name_or_path, device)
        self.task = task
        self.torch_dtype = torch_dtype or (torch.float16 if device == "cuda" else torch.float32)
        
        # Initialize pipeline
        # We use 'pipeline' for ease of implementation, but custom loops offer more control.
        # For a benchmark, pipeline is often robust enough if batch_size is handled.
        try:
            self.pipe = pipeline(
                task=task,
                model=model_name_or_path,
                device=0 if device == "cuda" else -1,
                torch_dtype=self.torch_dtype
            )
        except Exception as e:
            print(f"Error initializing pipeline for {model_name_or_path}: {e}")
            raise e

    def predict(self, input_text: Union[str, Dict[str, Any]]) -> Any:
        """
        Single sample prediction.
        """
        formatted_input = self._format_input(input_text)
        result = self.pipe(formatted_input)
        return self._postprocess(result)

    def batch_predict(self, inputs: List[Union[str, Dict[str, Any]]], batch_size: int = 32) -> List[Any]:
        """
        Batch prediction using the pipeline's built-in batching.
        """
        formatted_inputs = [self._format_input(x) for x in inputs]
        
        results = []
        # Pipeline handles batching via a generator or list
        # We pass the list directly which pipelien supports
        # Note: setting batch_size in pipe call
        
        # Using tqdm for progress if list is long
        iterator = self.pipe(formatted_inputs, batch_size=batch_size)
        
        # If inputs is very large, 'iterator' from pipe is a generator. We explicitly loop.
        for out in tqdm(iterator, total=len(formatted_inputs), desc="Inference", disable=len(formatted_inputs)<10):
            results.append(self._postprocess(out))
            
        return results

    def _format_input(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """
        Helper to flatten dict inputs (e.g. NLI hypothesis/premise) into string if model expects string.
        Models like BERT for NLI usually expect [SEP] separated text or specific formatting.
        However, the HF pipeline usually handles 'text_pair' args for some tasks, or we concat.
        
        For standard Text Classification pipeline: expects single string or (str, str).
        """
        if isinstance(input_data, str):
            return input_data
        
        if isinstance(input_data, dict):
            # NLI Standard: "premise [SEP] hypothesis" or similar, 
            # BUT many models are trained differently.
            # Best generic heuristic: "premise: ... hypothesis: ..." or space concatenation 
            # if specific formatting isn't enforced by a tokenizer wrapper.
            #
            # If the pipeline supports dictionary/args, we'd use that.
            # But standard 'text-classification' pipeline mainly takes text.
            # We will concat for now: "premise hypothesis"
            
            parts = []
            if "premise" in input_data: parts.append(input_data["premise"])
            if "hypothesis" in input_data: parts.append(input_data["hypothesis"])
            if "text" in input_data: parts.append(input_data["text"])
            
            return " ".join(parts)
            
        return str(input_data)

    def _postprocess(self, result: Any) -> Any:
        """
        Normalize output.
        Classification: return label or score.
        Generation: return generated text.
        """
        # Pipeline output format varies by task.
        # Classification: [{'label': 'LABEL_1', 'score': 0.99}] (list of dicts)
        # Generation: [{'generated_text': '...'}]
        
        if isinstance(result, list):
            item = result[0] # Usually list of 1 if input was single item
            if 'label' in item:
                return item['label']
            if 'generated_text' in item:
                return item['generated_text']
        
        return result
