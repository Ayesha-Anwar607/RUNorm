"""
Spelling-variant clustering for Roman Urdu.

Takes the top-N frequency vocabulary, groups words that are orthographic variants
of the same underlying Urdu lexeme, and proposes a canonical form for each cluster.

Clustering Architecture (Centroid Clustering with Linguistic Constraints):
1. Seed Anchor Clusters: Well-established irregular high-frequency Roman Urdu particles
   (e.g. hai/ha/hy, nahi/nhi/nai, kya/kia/kiya, mein/me/mai, etc.)
2. Centroid-based Grouping: Iterates through words in descending frequency order.
   The highest unassigned word forms a centroid.
3. Stringent Linguistic Boundary Guards:
   - 1-letter words are never merged with multi-letter words (avoids 'k' swallowing 'ka/ke/ko').
   - High-frequency inflectional and grammatical collisions (ka/ki/ke/ko, se/si, ne/ni, din/dil)
     are explicitly guarded.
   - Initial consonant phonology must match.
   - Length difference is tightly constrained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler

from runorm.normalizer.phonetic import urdu_phonetic_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Linguistic Boundary Guards: Pairs that must NEVER merge
# ---------------------------------------------------------------------------
FORBIDDEN_PAIRS: Set[frozenset] = {
    # Distinct postpositions & inflection markers
    frozenset({"ka", "ki"}),
    frozenset({"ka", "ke"}),
    frozenset({"ka", "ko"}),
    frozenset({"ke", "ki"}),
    frozenset({"ke", "ko"}),
    frozenset({"ki", "ko"}),
    frozenset({"k", "ka"}),
    frozenset({"k", "ki"}),
    frozenset({"k", "ke"}),
    frozenset({"k", "ko"}),
    frozenset({"se", "si"}),
    frozenset({"ne", "ni"}),
    # Lexical minimal pairs
    frozenset({"bhai", "bhi"}),
    frozenset({"bhai", "be"}),
    frozenset({"bhi", "be"}),
    frozenset({"kal", "kam"}),
    frozenset({"kar", "par"}),
    frozenset({"din", "dil"}),
    frozenset({"ab", "ap"}),
    frozenset({"hum", "tum"}),
    frozenset({"wo", "ye"}),
    frozenset({"tu", "to"}),
    frozenset({"to", "do"}),
    frozenset({"ja", "jo"}),
    frozenset({"jo", "jao"}),
    frozenset({"ha", "ho"}),
    frozenset({"na", "ni"}),
    frozenset({"na", "no"}),
    frozenset({"mein", "min"}),
    frozenset({"kah", "ka"}),
    frozenset({"ke", "keh"}),
}

# ---------------------------------------------------------------------------
# High-frequency Roman Urdu Anchor Equivalence Seeds
# (Maps irregular grammatical function words to their standardized canonical form)
# ---------------------------------------------------------------------------
SEED_CLUSTERS: List[Tuple[str, Set[str]]] = [
    ("hai", {"ha", "hai", "hy", "he", "h", "hia"}),
    ("mein", {"me", "mein", "mai", "men", "mn", "m"}),
    ("nahin", {"nahi", "nhi", "nai", "nahin", "nahe", "nh", "nhe"}),
    ("yeh", {"ye", "yeh", "yhi"}),
    ("woh", {"wo", "woh", "whi"}),
    ("kya", {"kya", "kia", "kiya"}),
    ("aur", {"or", "aur", "aor"}),
    ("bohat", {"bohat", "bahut", "bhot", "boht"}),
    ("bhi", {"bhi", "bi", "b"}),
    ("bhai", {"bhai", "bahi"}),
    ("acha", {"acha", "accha", "achha", "achaa"}),
    ("aap", {"ap", "aap", "app"}),
    ("phir", {"phir", "phr", "pir"}),
    ("matlab", {"mtlb", "matlab", "matlb"}),
    ("pehle", {"phly", "pehly", "pehle"}),
    ("samajh", {"samjh", "smj", "smjh", "samajh"}),
    ("mashallah", {"mashallah", "mashaallah"}),
    ("inshallah", {"inshallah", "inshaallah", "inshaaallah"}),
]

@dataclass
class ClusterRecord:
    cluster_id: int
    variant: str
    term_frequency: int
    document_frequency: int
    phonetic_key: str
    proposed_canonical: str
    confidence: str           # HIGH | MEDIUM | SINGLE
    approved_canonical: Optional[str] = None


def build_clusters(
    freq_df: pd.DataFrame,
    top_n: int = 1000,
) -> List[ClusterRecord]:
    """
    Build spelling-variant clusters from a vocabulary frequency DataFrame.
    """
    df = freq_df.head(top_n).copy()
    words: List[str] = df["word"].tolist()
    freq: Dict[str, int] = dict(zip(df["word"], df["term_frequency"]))
    doc_f: Dict[str, int] = dict(zip(df["word"], df["document_frequency"]))

    logger.info(f"Computing phonetic keys for {len(words)} words...")
    pk: Dict[str, str] = {w: urdu_phonetic_key(w) for w in words}

    assigned: Set[str] = set()
    clusters_data: List[Dict] = []

    # 1. First process SEED clusters
    for canon, seed_members in SEED_CLUSTERS:
        found = [w for w in words if w in seed_members]
        if found:
            clusters_data.append({
                "canonical": canon,
                "members": found,
                "confidence": "HIGH"
            })
            for w in found:
                assigned.add(w)

    # 2. Centroid-based clustering for all remaining words
    for w in words:
        if w in assigned:
            continue

        c_members = [w]
        assigned.add(w)
        pk_w = pk[w]
        len_w = len(w)

        for cand in words:
            if cand in assigned:
                continue

            # Check forbidden collision guards
            if frozenset({w, cand}) in FORBIDDEN_PAIRS:
                continue

            # Length guards: 1-letter words NEVER merge with other words
            if len_w == 1 or len(cand) == 1:
                continue

            # Initial consonant sound must align (c/k/q or w/v)
            if w[0] != cand[0]:
                if not ({w[0], cand[0]}.issubset({"k", "c", "q"}) or {w[0], cand[0]}.issubset({"w", "v"})):
                    continue

            pk_cand = pk[cand]
            lev = Levenshtein.distance(w, cand)
            jw = JaroWinkler.similarity(w, cand)

            # Rule A: Exact phonetic key match with minimal edit distance
            if pk_w == pk_cand and lev <= 2:
                c_members.append(cand)
                assigned.add(cand)
                continue

            # Rule B: Fuzzy match for longer words (length >= 4)
            if len_w >= 4 and len(cand) >= 4:
                if lev == 1 and jw >= 0.88:
                    c_members.append(cand)
                    assigned.add(cand)
                    continue
                if lev == 2 and jw >= 0.92 and abs(len_w - len(cand)) <= 1:
                    c_members.append(cand)
                    assigned.add(cand)
                    continue

        # Determine cluster confidence
        if len(c_members) == 1:
            conf = "SINGLE"
        elif any(pk_w == pk[m] for m in c_members[1:]):
            conf = "HIGH"
        else:
            conf = "MEDIUM"

        clusters_data.append({
            "canonical": w,
            "members": c_members,
            "confidence": conf
        })

    # Sort clusters by aggregate frequency descending
    clusters_data.sort(
        key=lambda c: -sum(freq[m] for m in c["members"])
    )

    records: List[ClusterRecord] = []
    for cluster_id, c in enumerate(clusters_data, start=1):
        canon = c["canonical"]
        conf = c["confidence"]
        # Sort members by frequency desc
        members_sorted = sorted(c["members"], key=lambda m: -freq[m])
        for variant in members_sorted:
            records.append(ClusterRecord(
                cluster_id=cluster_id,
                variant=variant,
                term_frequency=freq[variant],
                document_frequency=doc_f.get(variant, 0),
                phonetic_key=pk[variant],
                proposed_canonical=canon,
                confidence=conf
            ))

    total_clusters = len(clusters_data)
    multi_clusters = sum(1 for c in clusters_data if len(c["members"]) > 1)
    logger.info(
        f"Clustering complete: {total_clusters} clusters from {len(words)} words "
        f"({multi_clusters} multi-word variant clusters found)."
    )
    return records


def records_to_dataframe(records: List[ClusterRecord]) -> pd.DataFrame:
    """Convert ClusterRecord list to a reviewable DataFrame."""
    rows = [
        {
            "cluster_id": r.cluster_id,
            "variant": r.variant,
            "term_frequency": r.term_frequency,
            "document_frequency": r.document_frequency,
            "phonetic_key": r.phonetic_key,
            "proposed_canonical": r.proposed_canonical,
            "confidence": r.confidence,
            "approved_canonical": "",
        }
        for r in records
    ]
    return pd.DataFrame(rows)
