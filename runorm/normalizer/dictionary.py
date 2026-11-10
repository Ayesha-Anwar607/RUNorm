"""
Stage 1: Canonical Dictionary Lookup & Normalizer.

Loads the curated canonical dictionary (canonical_top1k.json) and provides
high-speed token normalization. Unrecognized words are passed through untouched
for downstream handling by Stage 2 (Phonetic clustering) and Stage 3 (Seq2Seq fallback).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class DictionaryNormalizer:
    """
    Stage 1 Normalizer: Exact dictionary mapping.
    Maps known Roman Urdu spelling variants to their standardized canonical spelling.
    """
    def __init__(self, dictionary_path: Path):
        self.dictionary_path = dictionary_path
        self.mapping: Dict[str, str] = {}
        self.load_dictionary()

    def load_dictionary(self) -> None:
        if not self.dictionary_path.exists():
            raise FileNotFoundError(f"Canonical dictionary not found: {self.dictionary_path}")

        with open(self.dictionary_path, "r", encoding="utf-8") as f:
            self.mapping = json.load(f)
        logger.info(f"Loaded {len(self.mapping):,} canonical mappings from {self.dictionary_path.name}")

    def normalize_token(self, token: str) -> Tuple[str, bool]:
        """
        Normalizes a single token.
        
        Args:
            token: The raw or cleaned token string.

        Returns:
            (normalized_token, is_in_dict)
            If found in dictionary, returns (canonical_word, True).
            Otherwise, returns (original_token, False).
        """
        tok_lower = token.lower().strip()
        if tok_lower in self.mapping:
            return self.mapping[tok_lower], True
        return token, False

    def __contains__(self, token: str) -> bool:
        return token.lower().strip() in self.mapping

    def __len__(self) -> int:
        return len(self.mapping)
