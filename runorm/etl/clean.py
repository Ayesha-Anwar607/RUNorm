"""
Cleaning module for RUNorm.

Applies language-agnostic cleaning rules driven by the language config.
All rules are ordered, and the original raw_text is NEVER mutated — this
module returns a cleaned copy while alignment information is preserved
by the downstream tokenizer.

Cleaning order (each step is optional, controlled by config):
  1.  Decode HTML entities & strip control characters
  2.  Strip URLs
  3.  Strip @mentions
  4.  Handle #hashtags (strip '#' prefix but keep the word)
  5.  Separate emojis from adjacent characters (insert spaces)
  6.  Separate punctuation from words (insert spaces around non-word chars)
  7.  Squash character elongations  (runs of >=N identical chars → N chars)
  8.  Normalize whitespace to single spaces
  9.  Lowercase (optional)
 10.  Strip leading / trailing whitespace
"""

import re
import html
import unicodedata
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex constants (language-agnostic, used by all languages)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|\S+@\S+\.\S+",
    re.IGNORECASE,
)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")  # group(1) keeps the word after '#'

# Unicode emoji range (covers most emoji blocks)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # Emoticons
    "\U0001F300-\U0001F5FF"   # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"   # Transport & Map
    "\U0001F700-\U0001F77F"   # Alchemical Symbols
    "\U0001F780-\U0001F7FF"   # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"   # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"   # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"   # Chess Symbols
    "\U0001FA70-\U0001FAFF"   # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"   # Dingbats
    "\U000024C2-\U0001F251"   # Enclosed characters
    "\U0000200D"               # Zero Width Joiner (part of emoji sequences)
    "\U0000FE0F"               # Variation Selector-16
    "]+",
    flags=re.UNICODE,
)

# Unicode control characters and invisible formatting marks
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"   # C0 controls (keep \t \n \r)
    r"\u200b-\u200f"                        # Zero-width, directional marks
    r"\u202a-\u202e"                        # Bidirectional overrides
    r"\u2060-\u206f"                        # Word joiner, etc.
    r"\ufeff"                               # BOM
    r"]"
)

# Multi-whitespace normalizer (also collapses \r\n etc.)
_WHITESPACE_RE = re.compile(r"\s+")

# Elongation squasher: matches any character repeated 3+ times
_ELONGATION_RE = re.compile(r"(.)\1{2,}", re.UNICODE)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def _strip_html_entities(text: str) -> str:
    """Decode HTML entities: &amp; → &, &lt; → <, &#39; → ' etc."""
    return html.unescape(text)


def _strip_control_characters(text: str) -> str:
    return _CONTROL_CHAR_RE.sub("", text)


def _strip_urls(text: str, placeholder: str = "") -> str:
    return _URL_RE.sub(placeholder, text)


def _strip_mentions(text: str, placeholder: str = "") -> str:
    return _MENTION_RE.sub(placeholder, text)


def _handle_hashtags(text: str, strip_symbol: bool = True) -> str:
    """Strip '#' but keep the hashtag text (e.g. '#PMLN' → 'PMLN')."""
    if strip_symbol:
        return _HASHTAG_RE.sub(r"\1", text)
    return text


def _space_emojis(text: str) -> str:
    """
    Insert a space before and after any emoji run so it doesn't
    concatenate with adjacent word characters.
    Example: "acha😂" → "acha 😂"
    """
    return _EMOJI_RE.sub(lambda m: f" {m.group(0)} ", text)


def _squash_elongations(text: str, max_repeat: int = 2) -> str:
    """
    Reduce runs of >=3 identical characters to `max_repeat` occurrences.
    Default max_repeat=2 so that valid Roman Urdu geminates are preserved
    (e.g., 'accha', 'abb') while clear elongations are squashed.
    Examples:
        'bohaaaat'  → 'bohaat'   (then dict stage maps to 'bohat')
        'sooooo'    → 'soo'
        'achhhh'    → 'achh'
        'accha'     → 'accha'    (unchanged — no run of 3+)
    """
    replacement = r"\1" * max_repeat
    return _ELONGATION_RE.sub(replacement, text)


def _separate_punctuation(text: str) -> str:
    """
    Insert a space between any word character and an adjacent non-word,
    non-space character so punctuation becomes its own token boundary.
    Examples:
        'theek?'    → 'theek ?'
        '...acha'   → '... acha'
        'han!'      → 'han !'
        'U.S.A'     → kept (dots between letters handled gracefully)
    """
    # Space before punctuation that immediately follows a word char
    text = re.sub(r"(\w)([^\w\s])", r"\1 \2", text)
    # Space after punctuation that immediately precedes a word char
    text = re.sub(r"([^\w\s])(\w)", r"\1 \2", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse any run of whitespace (space, tab, newline, etc.) to a single space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str, lang_config: Dict[str, Any]) -> str:
    """
    Apply all cleaning steps to a single raw text string, in order.
    Steps are individually toggled by the language config.

    Args:
        text:        Raw sentence string.
        lang_config: Parsed language YAML config dict (ur_roman.yaml).

    Returns:
        Cleaned text string with no mutation to the original.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cfg = lang_config.get("cleaning", {})
    sp_tokens = lang_config.get("special_tokens", {})

    t = text  # working copy

    # Step 1: HTML & control chars
    if cfg.get("strip_html_entities", True):
        t = _strip_html_entities(t)
    if cfg.get("strip_control_characters", True):
        t = _strip_control_characters(t)

    # Step 2: URLs — use an alphanumeric-safe placeholder so punctuation
    # separation (Step 6) won't split the angle-bracket tokens.
    # We replace with a no-punct placeholder internally, then restore after separation.
    _URL_PH = "URLTOKEN"
    _MENTION_PH = "MENTIONTOKEN"
    if cfg.get("strip_urls", True):
        t = _strip_urls(t, placeholder=_URL_PH)

    # Step 3: Mentions
    if cfg.get("strip_mentions", True):
        t = _strip_mentions(t, placeholder=_MENTION_PH)

    # Step 4: Hashtags — strip '#' symbol but keep the word
    if cfg.get("strip_hashtags", False) is False:
        t = _handle_hashtags(t, strip_symbol=True)

    # Step 5: Space out emojis (always; needed before punctuation separation)
    t = _space_emojis(t)

    # Step 6: Separate punctuation from word tokens
    if cfg.get("separate_punctuation", True):
        t = _separate_punctuation(t)

    # Step 7: Squash character elongations
    max_repeat = cfg.get("max_consecutive_char_repeat", 2)
    t = _squash_elongations(t, max_repeat=max_repeat)

    # Step 8: Normalize whitespace
    t = _normalize_whitespace(t)

    # Step 9: Lowercase (applied last so squashing/separation works case-aware)
    if cfg.get("lowercase", True):
        t = t.lower()

    # Restore final readable placeholders (after all transformations)
    final_url_token = sp_tokens.get("url", "<URL>")
    final_mention_token = sp_tokens.get("mention", "<MENTION>")
    t = t.replace("urltoken", final_url_token.lower() if cfg.get("lowercase", True) else final_url_token)
    t = t.replace("mentiontoken", final_mention_token.lower() if cfg.get("lowercase", True) else final_mention_token)

    return t
