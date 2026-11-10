import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

def reconstruct_whatsapp_split_row(row: pd.Series) -> str:
    """
    Reconstructs original message from rows where unescaped commas
    caused pandas to split the message across extra columns.
    Omits the trailing sentiment label if present.
    """
    vals = [str(x).strip() for x in row.values if pd.notna(x) and str(x).strip() != '']
    if not vals:
        return ""
    
    known_labels = {"neutral", "positive", "negative", "label", "sentiment"}
    # If the last item is a recognized sentiment/header label, drop it from raw text
    if len(vals) > 1 and vals[-1].lower() in known_labels:
        msg = ", ".join(vals[:-1])
    else:
        msg = ", ".join(vals)
    return msg

def load_and_map_corpus(file_path: Path, source_name: str, text_col: str = None, is_whatsapp_split: bool = False) -> pd.DataFrame:
    """
    Loads a single raw corpus CSV and standardizes it to (raw_text, source).
    Keeps raw text untouched.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    logger.info(f"Loading '{source_name}' from {file_path.name}...")
    df = pd.read_csv(file_path, low_memory=False)
    initial_count = len(df)

    if is_whatsapp_split:
        raw_texts = df.apply(reconstruct_whatsapp_split_row, axis=1)
    else:
        if text_col not in df.columns:
            raise KeyError(f"Expected column '{text_col}' not found in {file_path.name}. Found: {df.columns.tolist()}")
        raw_texts = df[text_col].astype(str)

    # Standardize schema
    result = pd.DataFrame({
        "raw_text": raw_texts,
        "source": source_name
    })

    # Drop null / strictly whitespace rows if any (logging count)
    valid_mask = result["raw_text"].notna() & (result["raw_text"].str.strip() != "")
    dropped = initial_count - valid_mask.sum()
    if dropped > 0:
        logger.warning(f"[{source_name}] Dropped {dropped} empty rows (out of {initial_count}).")

    result = result[valid_mask].copy()
    logger.info(f"[{source_name}] Loaded {len(result)} valid rows (original: {initial_count}).")
    return result

def merge_all_corpora(file_configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merges multiple raw corpus files according to configuration,
    standardizes schema to (id, raw_text, source), and assigns unique IDs.
    """
    dfs = []
    summary = []

    for cfg in file_configs:
        f_path = Path(cfg["path"])
        src_name = cfg["source_name"]
        t_col = cfg.get("text_column")
        is_wa_split = cfg.get("is_whatsapp_split", False)

        df_single = load_and_map_corpus(
            file_path=f_path,
            source_name=src_name,
            text_col=t_col,
            is_whatsapp_split=is_wa_split
        )
        dfs.append(df_single)
        summary.append({
            "source": src_name,
            "filename": f_path.name,
            "row_count": len(df_single)
        })

    merged_df = pd.concat(dfs, ignore_index=True)
    # Assign standard incremental 1-based ID
    merged_df.insert(0, "id", range(1, len(merged_df) + 1))

    return merged_df, pd.DataFrame(summary)
