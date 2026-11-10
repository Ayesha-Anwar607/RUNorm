"""
Vocabulary & frequency analysis module for RUNorm.

Streams the merged raw corpus in memory-efficient chunks, applies the full
clean → tokenize pipeline to every sentence, and builds aggregated frequency
tables. Outputs are written to data/interim/.

Public API:
    build_vocab(merged_csv_path, lang_config, chunk_size) -> VocabResult
"""

import logging
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from runorm.etl.clean import clean_text
from runorm.etl.tokenize import tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokens that should NEVER appear in the lexical vocabulary table.
# These are either structural placeholders or non-lexical symbols.
# ---------------------------------------------------------------------------
_PLACEHOLDER_TOKENS: Set[str] = {"<url>", "<mention>", "<emoji>", "<num>"}


# ---------------------------------------------------------------------------
# Data container returned by build_vocab()
# ---------------------------------------------------------------------------
@dataclass
class VocabResult:
    # Total sentences processed (after dropping empty)
    sentence_count: int = 0

    # Total token instances seen (before filtering noise)
    total_raw_token_count: int = 0

    # Total lexical token instances (after noise filter)
    total_lexical_token_count: int = 0

    # word -> total count across all sentences
    term_frequency: Counter = field(default_factory=Counter)

    # word -> number of distinct sentences containing it
    document_frequency: Counter = field(default_factory=Counter)

    # Count of words appearing exactly once (hapax legomena)
    hapax_count: int = 0

    # Vocabulary size (unique lexical words)
    vocab_size: int = 0


# ---------------------------------------------------------------------------
# Token filter
# ---------------------------------------------------------------------------

def _is_lexical(token_record: Dict[str, Any]) -> bool:
    """
    Returns True if a token record should count towards the lexical vocabulary.
    Filters out:
      - Punctuation tokens
      - Numeric tokens
      - Emoji tokens
      - Structural placeholders (<url>, <mention>, <emoji>)
      - Arabic-script tokens (kept in raw data but excluded from Roman Urdu vocab)
    """
    if token_record["is_punct"]:
        return False
    if token_record["is_numeric"]:
        return False
    if token_record["is_emoji"]:
        return False
    if token_record["is_arabic_script"]:
        return False
    if token_record["clean_token"].lower() in _PLACEHOLDER_TOKENS:
        return False
    return True


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_vocab(
    merged_csv_path: Path,
    lang_config: Dict[str, Any],
    chunk_size: int = 10_000,
    text_column: str = "raw_text",
) -> VocabResult:
    """
    Streams merged_csv_path in chunks, cleans and tokenizes every sentence,
    and aggregates term frequency and document frequency counters.

    Args:
        merged_csv_path: Path to data/interim/raw_merged.csv
        lang_config:     Parsed ur_roman.yaml config dict
        chunk_size:      Rows per chunk (tune for available RAM)
        text_column:     Name of the text column in the CSV

    Returns:
        A populated VocabResult dataclass.
    """
    if not merged_csv_path.exists():
        raise FileNotFoundError(f"Merged corpus not found: {merged_csv_path}")

    result = VocabResult()
    tf: Counter = Counter()      # term frequency
    df: Counter = Counter()      # document frequency

    total_rows_read = 0
    chunk_num = 0

    logger.info(f"Streaming '{merged_csv_path.name}' in chunks of {chunk_size:,}...")

    for chunk in pd.read_csv(merged_csv_path, chunksize=chunk_size, dtype=str):
        chunk_num += 1
        # Keep only the text column; drop rows where it is null/whitespace
        texts = chunk[text_column].dropna()
        texts = texts[texts.str.strip() != ""]
        total_rows_read += len(texts)

        for raw_text in texts:
            # --- Clean ---
            cleaned = clean_text(raw_text, lang_config)
            if not cleaned:
                continue

            # --- Tokenize ---
            token_records = tokenize(raw_text, cleaned, lang_config)
            result.total_raw_token_count += len(token_records)

            # --- Aggregate ---
            sentence_vocab: Set[str] = set()
            for tok in token_records:
                if not _is_lexical(tok):
                    continue
                word = tok["clean_token"]
                tf[word] += 1
                result.total_lexical_token_count += 1
                sentence_vocab.add(word)

            # Document frequency: each word counts once per sentence
            for word in sentence_vocab:
                df[word] += 1

        result.sentence_count = total_rows_read
        logger.info(f"  Chunk {chunk_num}: processed {total_rows_read:,} sentences so far...")

    result.term_frequency = tf
    result.document_frequency = df
    result.vocab_size = len(tf)
    result.hapax_count = sum(1 for c in tf.values() if c == 1)

    logger.info(
        f"Done. {result.sentence_count:,} sentences | "
        f"{result.vocab_size:,} unique words | "
        f"{result.hapax_count:,} hapax legomena"
    )
    return result


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def to_frequency_dataframe(result: VocabResult) -> pd.DataFrame:
    """
    Converts VocabResult counters into a ranked DataFrame.

    Columns: rank, word, term_frequency, document_frequency,
             tf_per_1k, df_ratio, word_length
    """
    rows = []
    for word, tf_count in result.term_frequency.most_common():
        df_count = result.document_frequency.get(word, 0)
        rows.append({
            "word": word,
            "term_frequency": tf_count,
            "document_frequency": df_count,
            "tf_per_1k": round(tf_count / result.total_lexical_token_count * 1000, 4),
            "df_ratio": round(df_count / result.sentence_count, 6),
            "word_length": len(word),
        })

    df = pd.DataFrame(rows)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def to_summary_dict(result: VocabResult) -> Dict[str, Any]:
    """Produces a JSON-serializable summary of corpus statistics."""
    return {
        "sentence_count": result.sentence_count,
        "total_raw_token_count": result.total_raw_token_count,
        "total_lexical_token_count": result.total_lexical_token_count,
        "vocab_size": result.vocab_size,
        "hapax_count": result.hapax_count,
        "hapax_ratio": round(result.hapax_count / result.vocab_size, 4) if result.vocab_size else 0,
        "avg_tokens_per_sentence": round(
            result.total_lexical_token_count / result.sentence_count, 2
        ) if result.sentence_count else 0,
        "top_20_words": [
            {"word": w, "freq": c}
            for w, c in result.term_frequency.most_common(20)
        ],
    }
