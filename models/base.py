from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class ModelRunner(ABC):
    """
    Abstract base class for model inference.
    Designed to support various backends (HuggingFace, API, etc.)
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        self.model_name = model_name_or_path
        self.device = device

    @abstractmethod
    def predict(self, input_text: Union[str, Dict[str, Any]]) -> Any:
        """
        Run inference on a single input.
        
        Args:
            input_text: Raw string or dictionary (e.g., {'premise': ..., 'hypothesis': ...})
            
        Returns:
            Prediction output (label, probabilities, or generation).
        """
        pass

    @abstractmethod
    def batch_predict(self, inputs: List[Union[str, Dict[str, Any]]], batch_size: int = 32) -> List[Any]:
        """
        Run inference on a batch of inputs.
        
        Args:
            inputs: List of strings or dictionaries.
            batch_size: Batch size for processing.
            
        Returns:
            List of prediction outputs.
        """
        pass
