"""
Standalone reproducibility script to execute the initial merge step.
Standardizes schemas from diverse CSV files to (id, raw_text, source).
Excludes foreign negative samples, reconstructs split WhatsApp messages,
and keeps raw_text untouched.
"""

import sys
import logging
from pathlib import Path
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runorm.etl.extract import merge_all_corpora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RUNorm-Merge")

def main():
    config_path = PROJECT_ROOT / "configs" / "pipeline.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to project root
    sources = config.get("sources", [])
    for src in sources:
        src["path"] = str(PROJECT_ROOT / src["path"])

    output_csv = PROJECT_ROOT / config.get("output", {}).get("interim_merged_csv", "data/interim/raw_merged.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting merge of raw corpora...")
    merged_df, summary_df = merge_all_corpora(sources)

    logger.info("Saving merged dataset...")
    merged_df.to_csv(output_csv, index=False, encoding="utf-8")

    logger.info("\n" + "="*50)
    logger.info("RAW MERGE SUMMARY REPORT")
    logger.info("="*50)
    for _, row in summary_df.iterrows():
        logger.info(f"- {row['source']:<25} ({row['filename']}): {row['row_count']:,} rows")
    logger.info("-" * 50)
    logger.info(f"Total rows in merged dataset: {len(merged_df):,}")
    logger.info(f"Output saved to: {output_csv}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
