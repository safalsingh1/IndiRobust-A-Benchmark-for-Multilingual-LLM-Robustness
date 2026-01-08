import random
from typing import Dict, Any, List

class CodeMixer:
    """
    Injects Code-Mixing (switching to English) into Hindi/Marathi text.
    Uses dictionary-based substitution for content words.
    """
    
    # Dictionary format: { 'lang_code': { 'native_word': 'english_word' } }
    # Using lower case for matching.
    MIXING_DICT = {
        'hi': {
            # Nouns (Objects/Concepts)
            "गाड़ी": "car", "घर": "house", "किताब": "book", "फोन": "phone",
            "स्कूल": "school", "बच्चा": "kid", "समय": "time", "दिमाग": "mind",
            "कहानी": "story", "सवाल": "question", "जवाब": "answer", "दोस्त": "friend",
            "प्यार": "love", "जिंदगी": "life", "दुनिया": "world", "ऑफिस": "office",
            # Adjectives
            "अच्छा": "good", "बुरा": "bad", "खुश": "happy", "नाराज": "angry",
            "मुश्किल": "difficult", "आसान": "easy", "जल्दी": "fast/early", "जरूरी": "important",
            # Verbs (Root/Common forms - substitution is tricky without stemming, exact match only here)
            "सोचना": "think", "करना": "do", "जाना": "go", "आना": "come",
            "देखना": "see", "समझना": "understand", "बोलना": "speak", "खेलना": "play",
             # Common connectives/fillers (sometimes mixed)
            "लेकिन": "but", "शायद": "maybe", "क्योंकि": "because"
        },
        'mr': {
            # Nouns
            "गाडी": "car", "घर": "house", "पुस्तक": "book", "शाळा": "school",
            "मित्र": "friend", "वेळ": "time", "प्रश्न": "question", "उत्तर": "answer",
            "जग": "world", "प्रेम": "love", "आयुष्य": "life",
            # Adjectives
            "चांगले": "good", "वाईट": "bad", "सोपे": "easy", "कठीण": "hard",
            "महत्वाचे": "important", "आनंदी": "happy",
            # Verbs (Exact forms mostly needed for simple lookup)
            "करणे": "do", "जाणे": "go", "येणे": "come", "पाहणे": "see"
        }
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(self, text: str, lang: str, mixing_ratio: float = 0.4) -> Dict[str, Any]:
        """
        Generate Code-Mixed text.
        
        Args:
            text: Input text (native script).
            lang: Language code ('hi', 'mr').
            mixing_ratio: Probability of replacing a known word with English.
            
        Returns:
            Dict with 'original_text', 'codemixed_text', 'language', 'perturbation_type'.
        """
        if not text:
            return self._wrap_result(text, text, lang)
            
        lang = lang.lower()
        if lang not in self.MIXING_DICT:
            # Fallback if unknown lang or 'en' (no code mix target defined for en->?)
            return self._wrap_result(text, text, lang)

        vocab = self.MIXING_DICT[lang]
        words = text.split()
        mixed_words = []
        
        for word in words:
            # Strip punctuation for lookup
            clean_word = word.strip(".,!?\"'।|")
            
            # Check exact match first
            if clean_word in vocab and random.random() < mixing_ratio:
                # Replace with first English equivalent (if multiple like fast/early, take first part)
                replacement = vocab[clean_word].split('/')[0]
                mixed_words.append(replacement)
            else:
                mixed_words.append(word)
        
        mixed_text = " ".join(mixed_words)
        return self._wrap_result(text, mixed_text, lang)

    def _wrap_result(self, original: str, mixed: str, lang: str) -> Dict[str, Any]:
        return {
            "original_text": original,
            "codemixed_text": mixed,
            "language": lang,
            "perturbation_type": "code-mixing"
        }

if __name__ == "__main__":
    cm = CodeMixer()
    print(cm.generate("मेरा घर बहुत अच्छा है।", "hi")) # My house very good is.
    print(cm.generate("हे पुस्तक खूप छान आहे.", "mr")) # This book very nice is.
