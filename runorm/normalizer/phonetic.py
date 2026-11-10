"""
Roman Urdu Phonetic Key Transformer.

Converts a Roman Urdu word into a normalised phonetic key so that
spelling variants that represent the same underlying Urdu word produce
the same (or near-identical) key.
"""

import re
from typing import Dict, List, Tuple

# Digraphs with 'h' for aspirated consonants (kept distinctive so 'bhai' != 'be')
_DIGRAPHS: List[Tuple[str, str]] = [
    ("kh", "Kh"),   # خ
    ("gh", "Gh"),   # غ
    ("ch", "Ch"),   # چ
    ("sh", "Sh"),   # ش
    ("ph", "Ph"),   # ف / پھ
    ("th", "Th"),   # تھ / ٹھ
    ("dh", "Dh"),   # دھ / ڈھ
    ("bh", "Bh"),   # بھ
    ("jh", "Jh"),   # جھ
    ("rh", "Rh"),   # ڑھ
]

def urdu_phonetic_key(word: str) -> str:
    """
    Return the phonetic key for a Roman Urdu word.
    Preserves aspirated consonant distinctions to avoid collisions like bhai vs bi/be.
    """
    if not word:
        return ""

    w = word.lower().strip()

    # Step 1: Consonant equivalences
    w = w.replace("v", "w").replace("q", "k")
    if "ch" not in w:
        w = w.replace("c", "k")

    # Step 2: Digraphs
    for src, tgt in _DIGRAPHS:
        w = w.replace(src, tgt)

    # Step 3: Vowel & Diphthong normalisation
    w = re.sub(r"iya|ya|ia", "Y", w)
    w = re.sub(r"aa+", "a", w)
    w = re.sub(r"ee+|ei|ie", "i", w)
    w = re.sub(r"oo+|ou", "u", w)
    w = re.sub(r"ai|ay|ey|ae", "e", w)
    w = re.sub(r"aw|ao|au", "o", w)

    # Trailing 'y' in Roman Urdu often functions as vowel 'e' (hy -> he, bachy -> bache)
    if len(w) > 1 and w.endswith("y"):
        w = w[:-1] + "e"

    # Silent trailing 'h' after a vowel (woh -> wo, yeh -> ye)
    if len(w) > 2 and w.endswith("h") and w[-2] in "aeiou":
        w = w[:-1]

    # Collapse consecutive identical characters
    w = re.sub(r"(.)\1+", r"\1", w)

    return w.upper()


def phonetic_edit_distance(key_a: str, key_b: str) -> int:
    """
    Levenshtein distance between two phonetic keys.
    """
    m, n = len(key_a), len(key_b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if key_a[i - 1] == key_b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
