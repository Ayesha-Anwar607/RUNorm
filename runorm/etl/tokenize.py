"""
Tokenizer module for RUNorm.

Converts a cleaned text string into a list of token records.
Each record preserves its positional alignment so that raw_token and
normalized_token can be stored side-by-side in the database.

Token record schema:
  {
      "position":          int,   # 0-indexed position within the sentence
      "raw_token":         str,   # original token as it appears in raw_text
      "clean_token":       str,   # lowercased / emoji-replaced cleaned form
      "is_punct":          bool,  # True if token is punctuation / symbol
      "is_numeric":        bool,  # True if token is entirely digits
      "is_emoji":          bool,  # True if token is an emoji cluster
      "is_arabic_script":  bool,  # True if token is in Arabic/Urdu script
  }
"""

import re
import unicodedata
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Script & character classifiers
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"^\d+$")
_PUNCT_RE = re.compile(r"^[^\w\s]+$", re.UNICODE)

# Arabic / Urdu script Unicode block: U+0600 – U+06FF
_ARABIC_RE = re.compile(r"^[\u0600-\u06FF]+$")

# Emoji detector: same ranges as in clean.py
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def _is_emoji(token: str) -> bool:
    return bool(_EMOJI_RE.fullmatch(token))


def _is_arabic_script(token: str) -> bool:
    return bool(_ARABIC_RE.fullmatch(token))


def _is_numeric(token: str) -> bool:
    return bool(_DIGIT_RE.fullmatch(token))


def _is_punct(token: str) -> bool:
    # Pure punctuation: non-word, non-whitespace only
    return bool(_PUNCT_RE.fullmatch(token)) and not _is_emoji(token)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(
    raw_text: str,
    cleaned_text: str,
    lang_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Tokenizes a cleaned sentence into a list of structured token records.

    We split the *cleaned* text into tokens, then try to align each
    cleaned token back to its counterpart in the *raw* text so that
    raw_token is preserved. If alignment fails (edge cases), raw_token
    falls back to the cleaned form.

    Args:
        raw_text:     Original uncleaned sentence from the corpus.
        cleaned_text: Output of clean.clean_text(raw_text, lang_config).
        lang_config:  Parsed language YAML config dict.

    Returns:
        List of token dicts, one per token.
    """
    cfg = lang_config.get("tokenization", {})
    sp_tokens = lang_config.get("special_tokens", {})
    emoji_placeholder = sp_tokens.get("emoji_placeholder", "<EMOJI>")
    min_len = cfg.get("min_token_length", 1)

    # Split cleaned text on whitespace to get token strings
    raw_split = raw_text.split() if raw_text else []
    cleaned_tokens = cleaned_text.split() if cleaned_text else []

    # Build an alignment index from the raw text for best-effort raw_token lookup
    # We walk raw_split and cleaned_tokens in parallel (they may differ in count
    # due to punctuation separation and emoji spacing inserted during cleaning).
    # Strategy: for each cleaned token, find the best matching raw word.
    raw_token_map = _build_raw_alignment(raw_split, cleaned_tokens)

    records = []
    for pos, clean_tok in enumerate(cleaned_tokens):
        if len(clean_tok) < min_len:
            continue

        emoji_flag = _is_emoji(clean_tok)
        arabic_flag = _is_arabic_script(clean_tok)
        numeric_flag = _is_numeric(clean_tok)
        punct_flag = _is_punct(clean_tok)

        # Determine the clean_token value for vocab building
        if emoji_flag:
            vocab_token = emoji_placeholder
        else:
            vocab_token = clean_tok  # already lowercased by cleaner

        raw_tok = raw_token_map.get(pos, clean_tok)

        records.append({
            "position": pos,
            "raw_token": raw_tok,
            "clean_token": vocab_token,
            "is_punct": punct_flag,
            "is_numeric": numeric_flag,
            "is_emoji": emoji_flag,
            "is_arabic_script": arabic_flag,
        })

    return records


def _build_raw_alignment(
    raw_words: List[str], cleaned_tokens: List[str]
) -> Dict[int, str]:
    """
    Best-effort positional alignment from cleaned tokens back to raw words.
    Returns a dict mapping cleaned_token_position → raw_word.

    Strategy:
      - Walk both lists with two pointers.
      - For each cleaned token, check if the current raw word contains it
        (accounting for punctuation separation and elongation squashing).
      - If yes, store the mapping and advance the raw pointer.
      - If no, store the cleaned token itself as a fallback.
    """
    mapping: Dict[int, str] = {}
    r_ptr = 0
    n_raw = len(raw_words)

    for c_idx, c_tok in enumerate(cleaned_tokens):
        if r_ptr >= n_raw:
            mapping[c_idx] = c_tok
            continue

        raw_word = raw_words[r_ptr]
        # Check membership: does the cleaned token appear inside the raw word
        # (after lowercasing the raw word for comparison)?
        if c_tok in raw_word.lower():
            mapping[c_idx] = raw_word
            r_ptr += 1
        else:
            # Punctuation that was separated, or emoji that was spaced out —
            # it won't match a whole raw word cleanly. Use cleaned as fallback.
            mapping[c_idx] = c_tok
            # Do NOT advance r_ptr — the raw word is still pending

    return mapping


# ---------------------------------------------------------------------------
# Sentence-level convenience wrapper
# ---------------------------------------------------------------------------

def tokenize_sentence(
    raw_text: str,
    lang_config: Dict[str, Any],
    cleaner_fn=None,
) -> Dict[str, Any]:
    """
    Full pipeline for a single sentence:
      raw_text → clean → tokenize → token records

    Args:
        raw_text:   Original sentence string.
        lang_config: Language config dict.
        cleaner_fn: Optionally inject a custom cleaning function;
                    defaults to runorm.etl.clean.clean_text.

    Returns:
        {
            "raw_text":     str,
            "clean_text":   str,
            "tokens":       List[token_record],
            "token_count":  int,
        }
    """
    if cleaner_fn is None:
        from runorm.etl.clean import clean_text as cleaner_fn

    cleaned = cleaner_fn(raw_text, lang_config)
    tokens = tokenize(raw_text, cleaned, lang_config)

    return {
        "raw_text": raw_text,
        "clean_text": cleaned,
        "tokens": tokens,
        "token_count": len(tokens),
    }
