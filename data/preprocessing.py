import unicodedata
import re
from typing import Dict, Any

def normalize_text(text: str) -> str:
    """
    Normalize text:
    - Unicode NFKC normalization
    - Remove replacement characters
    - Collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Remove control characters or replacement chars if needed
    text = text.replace("\ufffd", "")
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_example(example: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """
    Apply preprocessing to a standardized example.
    Modifies text fields in-place.
    """
    if task_type == "classification":
        if "text" in example:
            example["text"] = normalize_text(example["text"])
            
    elif task_type == "nli":
        if "premise" in example:
            example["premise"] = normalize_text(example["premise"])
        if "hypothesis" in example:
            example["hypothesis"] = normalize_text(example["hypothesis"])
            
    return example
