import random
from typing import Dict, List, Any

class Paraphraser:
    """
    Rule-based Paraphraser using synonym substitution.
    Avoids external APIs.
    """
    
    # Simple hardcoded synonym dictionary for demonstration & reproducibility
    # In a real expanded version, this would load from a file or WordNet.
    SYNONYMS = {
        'en': {
            "good": ["nice", "excellent", "great", "fine"],
            "bad": ["terrible", "awful", "poor", "negative"],
            "happy": ["joyful", "glad", "content", "cheerful"],
            "sad": ["unhappy", "sorrowful", "depressed", "down"],
            "big": ["large", "huge", "massive", "giant"],
            "small": ["tiny", "little", "miniature", "minor"],
            "verify": ["check", "validate", "confirm", "test"],
            "robust": ["strong", "sturdy", "resilient", "tough"],
            "person": ["human", "individual", "someone", "man/woman"],
            "classification": ["categorization", "grouping", "sorting"],
            "inference": ["deduction", "conclusion", "reasoning"],
            "text": ["content", "script", "writing", "passage"]
        },
        'hi': {
            "अच्छा": ["बढ़िया", "उत्तम", "श्रेष्ठ", "लाजवाब"], # Good
            "बुरा": ["खराब", "गलत", "बेकार", "घटिया"], # Bad
            "खुश": ["प्रसन्न", "आनंदित", "हर्षित", "मगन"], # Happy
            "दुखी": ["उदास", "परेशान", "व्यथित", "खिन्न"], # Sad
            "बड़ा": ["विशाल", "भारी", "महान", "अहम"], # Big
            "छोटा": ["लघु", "साधारण", "तुच्छ", "कम"], # Small
            "स्वागत": ["अभिनंदन", "सत्कार", "आदर", "इस्तकबाल"], # Welcome
            "भाषा": ["बोली", "ज़बान", "वाणी", "माध्यम"], # Language
            "परीक्षा": ["इम्तिहान", "जांच", "परख", "मूल्यांकन"] # Test/Exam
        },
        # Add 'mr' and 'bn' placeholder or basic maps as needed
        'mr': {
             "चांगले": ["छान", "उत्तम", "बढ़िया"],
             "वाईट": ["खराब", "चुकीचे"],
        },
        'bn': {
             "ভালো": ["উত্তম", "চমৎকার", "সুন্দর"],
             "খারাপ": ["বাজে", "মন্দ", "অশুভ"]
        }
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, text: str, lang: str = 'en', strategy: str = 'synonym', substitution_rate: float = 0.3) -> Dict[str, Any]:
        """
        Generate a paraphrase.
        
        Args:
            text: Input text.
            lang: Language code ('en', 'hi', 'mr', 'bn').
            strategy: Perturbation strategy (currently only 'synonym').
            substitution_rate: Probability of replacing a word if a synonym exists.
            
        Returns:
            Dictionary with metadata.
        """
        if not text:
            return self._wrap_result(text, text, lang, strategy)

        lang = lang.lower()
        if lang not in self.SYNONYMS:
            # Fallback to En or just return original if completely unknown
            # If lang is 'hindi' etc normalize it.
            if lang.startswith('hi'): lang = 'hi'
            elif lang.startswith('en'): lang = 'en'
            elif lang.startswith('mr'): lang = 'mr'
            elif lang.startswith('bn'): lang = 'bn'
        
        paraphrased_text = text
        if strategy == 'synonym':
            paraphrased_text = self._synonym_substitution(text, lang, substitution_rate)
            
        return self._wrap_result(text, paraphrased_text, lang, strategy)

    def _synonym_substitution(self, text: str, lang: str, rate: float) -> str:
        if lang not in self.SYNONYMS:
            return text
            
        words = text.split()
        new_words = []
        lang_syns = self.SYNONYMS[lang]
        
        for word in words:
            # Simple cleaning for lookup (remove punctuation for key check)
            # This is a basic implementation.
            clean_word = word.strip(".,!?\"'")
            if clean_word in lang_syns and random.random() < rate:
                replacement = random.choice(lang_syns[clean_word])
                # Attempt to preserve punctuation?? 
                # For this basic version, we just replace the word. 
                # A fancier version would re-attach punctuation.
                # Let's simple-replace for now.
                new_words.append(replacement)
            else:
                new_words.append(word)
                
        return " ".join(new_words)

    def _wrap_result(self, original: str, paraphrased: str, lang: str, perturbation_type: str) -> Dict[str, Any]:
        return {
            "original_text": original,
            "paraphrased_text": paraphrased,
            "language": lang,
            "perturbation_type": perturbation_type
        }

if __name__ == "__main__":
    # Quick sanity check
    p = Paraphraser()
    print(p.generate("This is a good verification test.", "en"))
    print(p.generate("यह एक अच्छा दिन है।", "hi"))
