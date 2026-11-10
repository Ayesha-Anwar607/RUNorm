"""
Runner script: Build vocabulary frequency table from the merged raw corpus.

Reads:   data/interim/raw_merged.csv
Writes:
  data/interim/vocab_frequency.csv      -- all words ranked by frequency
  data/interim/top_1000_candidates.csv  -- top 1000 words for dict curation
  data/interim/frequency_summary.json   -- corpus-level statistics

Usage:
  python scripts/build_vocab_frequency.py
"""

import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runorm.utils.config import load_language_config
from runorm.etl.vocab import build_vocab, to_frequency_dataframe, to_summary_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RUNorm-Vocab")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MERGED_CSV   = PROJECT_ROOT / "data" / "interim" / "raw_merged.csv"
OUT_FREQ_CSV = PROJECT_ROOT / "data" / "interim" / "vocab_frequency.csv"
OUT_TOP_CSV  = PROJECT_ROOT / "data" / "interim" / "top_1000_candidates.csv"
OUT_SUMMARY  = PROJECT_ROOT / "data" / "interim" / "frequency_summary.json"


def main():
    logger.info("Loading language config: ur_roman")
    lang_cfg = load_language_config("ur_roman")

    # --- Run the vocab builder ---
    result = build_vocab(
        merged_csv_path=MERGED_CSV,
        lang_config=lang_cfg,
        chunk_size=10_000,
        text_column="raw_text",
    )

    # --- Export 1: Full ranked vocabulary ---
    freq_df = to_frequency_dataframe(result)
    freq_df.to_csv(OUT_FREQ_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved full vocab table ({len(freq_df):,} words) → {OUT_FREQ_CSV.name}")

    # --- Export 2: Top 1000 candidates for hand-curation ---
    top_df = freq_df.head(1000).copy()
    top_df.to_csv(OUT_TOP_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved top-1000 candidates → {OUT_TOP_CSV.name}")

    # --- Export 3: JSON summary ---
    summary = to_summary_dict(result)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved frequency summary → {OUT_SUMMARY.name}")

    # --- Print key stats to terminal ---
    logger.info("")
    logger.info("=" * 55)
    logger.info("VOCABULARY ANALYSIS SUMMARY")
    logger.info("=" * 55)
    logger.info(f"  Sentences processed  : {result.sentence_count:,}")
    logger.info(f"  Total tokens (raw)   : {result.total_raw_token_count:,}")
    logger.info(f"  Lexical tokens       : {result.total_lexical_token_count:,}")
    logger.info(f"  Unique vocab size    : {result.vocab_size:,}")
    logger.info(f"  Hapax legomena       : {result.hapax_count:,} "
                f"({100*result.hapax_count/result.vocab_size:.1f}% of vocab)")
    logger.info(f"  Avg tokens/sentence  : "
                f"{result.total_lexical_token_count/result.sentence_count:.1f}")
    logger.info("-" * 55)
    logger.info("  TOP 20 WORDS:")
    for w, c in result.term_frequency.most_common(20):
        bar = "█" * min(30, int(c / result.term_frequency.most_common(1)[0][1] * 30))
        logger.info(f"    {w:<20} {c:>7,}  {bar}")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
