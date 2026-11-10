"""
Exporter script for Stage 1 Canonical Dictionary (canonical_top1k.json).

Applies:
1. Core Anchor Mappings (high-frequency irregular particles -> canonical).
2. Inflection Disambiguation:
   Preserves grammatical gender/number inflections:
   - Masculine singular (-a, e.g. apna, mera, karta, raha, hota)
   - Feminine singular (-i, e.g. apni, meri, karti, rahi, hoti)
   - Masculine plural / oblique (-e / -y, e.g. apne/apny, mere/mery, karte/krty, rahe/rahy, hote/hoty)
   - Plural oblique (-on / -o, e.g. walo, logon)
3. Spurious minimal-pair separations (e.g. 'rakha' != 'raha', 'beta' != 'beti').
4. Exports JSON to lang_packs/ur_roman/dictionaries/canonical_top1k.json
"""

import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RUNorm-ExportDict")

REVIEW_CSV = PROJECT_ROOT / "data" / "interim" / "clusters_for_review.csv"
OUTPUT_JSON = PROJECT_ROOT / "lang_packs" / "ur_roman" / "dictionaries" / "canonical_top1k.json"
OUTPUT_CSV = PROJECT_ROOT / "lang_packs" / "ur_roman" / "dictionaries" / "canonical_top1k.csv"

# Irregular high-frequency function particles exempt from ending-based inflection splitting
EXEMPT_CANONICALS = {
    "hai", "nahin", "yeh", "woh", "aap", "bhi", "bhai", "acha",
    "aur", "bohat", "phir", "pehle", "kya", "matlab", "samajh",
    "mashallah", "inshallah", "allah"
}

# Explicit word-level corrections and standardizations
EXPLICIT_CORRECTIONS = {
    "rakha": "rakha",
    "kahan": "kahan",
    "kahi": "kahi",
    "kaha": "kaha",
    "apko": "apko",
    "apka": "apka",
    "apki": "apki",
    "iska": "iska",
    "isko": "isko",
    "iski": "iski",
    "yaha": "yahan",
    "yahan": "yahan",
    "waha": "wahan",
    "wahan": "wahan",
    "beta": "beta",
    "beti": "beti",
    "rha": "raha",
    "rhi": "rahi",
    "rhy": "rahe",
    "rahy": "rahe",
    "rahe": "rahe",
    "skta": "sakta",
    "skti": "sakti",
    "skty": "sakte",
    "sakty": "sakte",
    "sakte": "sakte",
    "krta": "karta",
    "krti": "karti",
    "krty": "karte",
    "krte": "karte",
    "karte": "karte",
    "krna": "karna",
    "krni": "karni",
    "krny": "karne",
    "krne": "karne",
    "karne": "karne",
    "apny": "apne",
    "mery": "mere",
    "tery": "tere",
    "hoty": "hote",
    "jaty": "jate",
    "dety": "dete",
    "kehty": "kehte",
    "waly": "wale",
    "sary": "sare",
    "jany": "jane",
    "lety": "lete",
    "bachy": "bache",
    "aisy": "aise",
    "mily": "mile",
    "hony": "hone",
    "hone": "hone",
    "honi": "honi",
    "hona": "hona",
    "itny": "itne",
    "kitny": "kitne",
    "kese": "kaise",
    "kesy": "kaise",
    "kesa": "kaisa",
    "kesi": "kaisi",
    "jese": "jaise",
    "jesy": "jaise",
    "jesa": "jaisa",
    "thora": "thora",
    "thori": "thori",
    "bara": "bara",
    "bari": "bari",
    "bare": "bare",
    "bary": "bare",
}

def get_inflection_slot(w: str) -> str:
    w = w.lower()
    if w.endswith("i") or w.endswith("ee"):
        return "FEM_SG"
    if w.endswith("on") or (w.endswith("o") and len(w) > 3 and not w.endswith("ko") and not w.endswith("do") and not w.endswith("kro")):
        return "PL_OBL"
    if w.endswith("e") or (w.endswith("y") and len(w) >= 3 and w not in {"kya", "kia", "bhai", "nahi", "kahi", "sahi"}):
        return "MASC_PL"
    if w.endswith("a") or w.endswith("aa"):
        return "MASC_SG"
    return "BASE"


def choose_subcluster_canonical(variants: list, freq_map: dict) -> str:
    """
    Selects the cleanest, most standard canonical form for a subcluster of variants.
    Prefers non-abbreviated, full-vowel standard transliterations.
    """
    # Otherwise check if explicit corrections exist for any variant
    # But only if the variant itself matches the explicit correction!
    # For example, 'rakha' -> 'rakha', but 'raha' -> 'raha'
    for v in variants:
        if v in EXPLICIT_CORRECTIONS and v == EXPLICIT_CORRECTIONS[v]:
            pass

    # Sort by frequency descending, but prioritize full-vowel standard forms
    # over abbreviated informal forms (e.g. 'karta' > 'krta', 'karna' > 'krna', 'sakta' > 'skta')
    full_vowel_pairs = {
        "krta": "karta",
        "krna": "karna",
        "skta": "sakta",
        "rha": "raha",
    }
    for abbrev, full in full_vowel_pairs.items():
        if full in variants:
            return full

    sorted_vars = sorted(variants, key=lambda x: -freq_map.get(x, 0))
    top_var = sorted_vars[0]

    # If top_var ends with informal 'y' (e.g. apny, mery), standardize to 'e'
    if top_var.endswith("y") and len(top_var) > 2 and get_inflection_slot(top_var) == "MASC_PL":
        return top_var[:-1] + "e"

    return top_var


def build_canonical_mapping() -> tuple[dict, pd.DataFrame]:
    if not REVIEW_CSV.exists():
        raise FileNotFoundError(f"Missing review CSV: {REVIEW_CSV}")

    df = pd.read_csv(REVIEW_CSV)
    freq_map = dict(zip(df["variant"], df["term_frequency"]))
    mapping = {}
    rows_for_df = []

    for cid, group in df.groupby("cluster_id"):
        canon = group["proposed_canonical"].iloc[0]
        variants = group["variant"].tolist()

        # If it is an exempt core particle (e.g. hai, nahin, yeh, aur, bohat), keep unified
        if canon in EXEMPT_CANONICALS:
            for v in variants:
                target = EXPLICIT_CORRECTIONS.get(v, canon)
                mapping[v] = target
                rows_for_df.append({
                    "cluster_id": cid,
                    "variant": v,
                    "canonical": target,
                    "term_frequency": freq_map.get(v, 0),
                    "inflection_class": "CORE_PARTICLE"
                })
            continue

        # Check if cluster contains multiple inflection slots
        slots = defaultdict(list)
        for v in variants:
            s = get_inflection_slot(v)
            slots[s].append(v)

        has_gender_inflection = (
            len(slots) > 1 and 
            ("FEM_SG" in slots or "MASC_PL" in slots) and 
            ("MASC_SG" in slots or "BASE" in slots or "MASC_PL" in slots)
        )

        if has_gender_inflection:
            # Separate by grammatical slot
            for slot_name, slot_vars in slots.items():
                slot_canon = choose_subcluster_canonical(slot_vars, freq_map)
                for v in slot_vars:
                    target = EXPLICIT_CORRECTIONS.get(v, slot_canon)
                    mapping[v] = target
                    rows_for_df.append({
                        "cluster_id": cid,
                        "variant": v,
                        "canonical": target,
                        "term_frequency": freq_map.get(v, 0),
                        "inflection_class": slot_name
                    })
        else:
            # Single inflection or regular cluster
            sub_canon = choose_subcluster_canonical(variants, freq_map)
            for v in variants:
                target = EXPLICIT_CORRECTIONS.get(v, sub_canon)
                mapping[v] = target
                rows_for_df.append({
                    "cluster_id": cid,
                    "variant": v,
                    "canonical": target,
                    "term_frequency": freq_map.get(v, 0),
                    "inflection_class": "REGULAR"
                })

    out_df = pd.DataFrame(rows_for_df)
    return mapping, out_df


def main():
    logger.info("Building canonical dictionary with inflection preservation...")
    mapping, out_df = build_canonical_mapping()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON dictionary (lookup engine)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved canonical dictionary ({len(mapping):,} entries) -> {OUTPUT_JSON}")

    # Save CSV table (documentation & review)
    out_df.sort_values(by=["term_frequency"], ascending=False, inplace=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info(f"Saved canonical table -> {OUTPUT_CSV}")

    # Log verification summary
    logger.info("\n" + "="*60)
    logger.info("CANONICAL DICTIONARY EXPORT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total vocabulary variants mapped: {len(mapping):,}")
    logger.info(f"Unique canonical target words   : {len(set(mapping.values())):,}")
    
    # Test specific key inflection sets in logs:
    test_keys = [
        "ha", "hy", "hai", "he",
        "nahi", "nhi", "nai", "nahin",
        "raha", "rahi", "rahe", "rahy",
        "apna", "apni", "apne", "apny",
        "karta", "karti", "karte", "krty",
        "bhai", "bahi", "bhi", "bi"
    ]
    logger.info("\nSample Mappings:")
    for k in test_keys:
        if k in mapping:
            logger.info(f"  {k:<10} -> {mapping[k]}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
