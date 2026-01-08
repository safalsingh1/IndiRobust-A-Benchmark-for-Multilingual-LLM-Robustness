import random
import unicodedata

class CharNoiseInjector:
    """
    Injects character-level noise into text.
    Supports: Random Deletion, Swap, Vowel Dropping.
    """
    
    # Language-specific vowel definitions (Independent Vowels + Matras/Diacritics)
    VOWELS = {
        'en': set("aeiouAEIOU"),
        'hi': set([
            # Devanagari Independent Vowels
            '\u0905', '\u0906', '\u0907', '\u0908', '\u0909', '\u090a', '\u090b', '\u090f', '\u0910', '\u0913', '\u0914',
            # Devanagari Matras
            '\u093e', '\u093f', '\u0940', '\u0941', '\u0942', '\u0943', '\u0947', '\u0948', '\u094b', '\u094c',
            # Chandrabindu/Anusvara (optional, often treated as vowel-like modifiers)
            '\u0901', '\u0902', '\u0903'
        ]),
        'mr': set([
            # Marathi uses Devanagari (mostly same as Hindi plus specific Marathi vowels like \u0935 \u0938 etc? No, \u0905-\u0914 mostly)
            # Reusing Hindi set + standard Marathi specific if strictly needed, but broad Devanagari coverage is usually sufficient.
            '\u0905', '\u0906', '\u0907', '\u0908', '\u0909', '\u090a', '\u090b', '\u090f', '\u0910', '\u0913', '\u0914',
            '\u093e', '\u093f', '\u0940', '\u0941', '\u0942', '\u0943', '\u0947', '\u0948', '\u094b', '\u094c',
            '\u0901', '\u0902', '\u0903',
            # Marathi 'L' vowel
            '\u090c', '\u0962', '\u0963'
        ]),
        'bn': set([
            # Bengali Independent Vowels
            '\u0985', '\u0986', '\u0987', '\u0988', '\u0989', '\u098a', '\u098b', '\u098f', '\u0990', '\u0993', '\u0994',
            # Bengali Matras
            '\u09be', '\u09bf', '\u09c0', '\u09c1', '\u09c2', '\u09c3', '\u09c7', '\u09c8', '\u09cb', '\u09cc',
             '\u0981', '\u0982', '\u0983'
        ])
    }
    
    # Fallback for Marathi to use Hindi set if close enough, but explicit is better.
    # Note: 'mr' set above covers standard Devanagari range.

    @staticmethod
    def random_deletion(text: str, noise_level: float = 0.1) -> str:
        """
        Randomly delete characters with probability `noise_level`.
        """
        if not text: return ""
        if noise_level <= 0: return text
        
        return "".join([c for c in text if random.random() > noise_level])

    @staticmethod
    def random_swap(text: str, noise_level: float = 0.1) -> str:
        """
        Swap adjacent characters with probability `noise_level`.
        """
        if not text: return ""
        if noise_level <= 0: return text
        
        chars = list(text)
        n = len(chars)
        for i in range(n - 1):
            if random.random() < noise_level:
                # Swap
                chars[i], chars[i+1] = chars[i+1], chars[i]
                # Skip next to avoid re-swapping immediately (simple heuristic)
                # But simple loop is fine for "random noise"
                
        return "".join(chars)

    @classmethod
    def vowel_drop(cls, text: str, noise_level: float = 0.1, lang: str = 'en') -> str:
        """
        Drop vowels with probability `noise_level`.
        Language-aware: uses specific vowel sets for En, Hi, Mr, Bn.
        """
        if not text: return ""
        if noise_level <= 0: return text
        
        lang = lang.lower()
        if lang not in cls.VOWELS:
            # Fallback or strict error? 
            # For robustness, fallback to 'en' or empty set (no drop)
            # Let's fallback to no-op for unknown lang to avoid destroying text
            if lang in ['hindi', 'hind']: lang = 'hi'
            elif lang in ['marathi']: lang = 'mr'
            elif lang in ['bengali', 'bangla']: lang = 'bn'
            elif lang in ['english']: lang = 'en'
            else:
                return text # Unknown language, safe exit
        
        vowels = cls.VOWELS[lang]
        
        result = []
        for char in text:
            if char in vowels and random.random() < noise_level:
                continue
            result.append(char)
            
        return "".join(result)

    @classmethod
    def inject_noise(cls, text: str, noise_types: list = ['delete'], noise_level: float = 0.1, lang: str = 'en') -> str:
        """
        Apply multiple noise types in sequence.
        """
        perturbed_text = text
        for n_type in noise_types:
            if n_type == 'delete':
                perturbed_text = cls.random_deletion(perturbed_text, noise_level)
            elif n_type == 'swap':
                perturbed_text = cls.random_swap(perturbed_text, noise_level)
            elif n_type == 'vowel_drop':
                perturbed_text = cls.vowel_drop(perturbed_text, noise_level, lang)
        
        return perturbed_text
