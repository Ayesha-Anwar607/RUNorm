"""
Manual smoke tests for the cleaning and tokenization pipeline.
Runs against hand-crafted tricky cases AND real samples from the merged corpus.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runorm.utils.config import load_language_config
from runorm.etl.clean import clean_text
from runorm.etl.tokenize import tokenize_sentence

LANG_CFG = load_language_config("ur_roman")

# ---------------------------------------------------------------------------
# Curated test cases covering each cleaning rule
# ---------------------------------------------------------------------------
TESTS = [
    # (description, raw_text)
    ("Basic informal",         "kah dy parh rha"),
    ("Uppercase mixed",        "TaTe bhi"),
    ("Elongation squash",      "Yaaar ye movieeee seriously bohat achi thi"),
    ("Emoji attached",         "thank you sir 😊"),
    ("Emoji glued to word",    "acha😂 bht maza aya"),
    ("Emoji chain",            "🙈🙃🤣 lol"),
    ("Mention & URL",          "follow @user at https://t.co/xyz for more"),
    ("Hashtag",                "pmln #PMLN zindabad"),
    ("Punctuation attached",   "theek? han! wah."),
    ("Newline/carriage return","Okay!\r\nWo abhi welcome keh rai!😂"),
    ("HTML entity",            "acha &amp; theek ha &lt;3"),
    ("Elongation geminate ok", "accha abb challo"),           # these should NOT squash
    ("Elongation 3+ squash",   "sooooo bohaaaat achhhh"),    # should squash to 2
    ("Arabic script mixed",    "Masha Allah ماشاءاللہ bohat khush hon"),
    ("Toxic real sample",      "o bhosdke waqt maryam he puri pmln kanjra kithy shafqat krda rya"),
    ("Numbers mixed",          "just 5 winning Trade chye na apko"),
    ("Dots/ellipsis",          "Usi waqt tak in ka kaam mehaz shauqia samjha ja raha tha..."),
    ("Extra spaces",           " Abhi bari hojao phir siyasat ki baat karna "),
    ("Real – multi emoji",     "Okay!\r\nWo abhi welcome keh rai!😂"),
    ("Real – long sentence",   "bhai pmln national action plan bnaya jis terrorism khatm or rangers itny akhtiyar diya k punjab operation skty"),
]


def run_tests():
    sep = "-" * 70
    for desc, raw in TESTS:
        result = tokenize_sentence(raw, LANG_CFG)
        print(sep)
        print(f"[{desc}]")
        print(f"  RAW   : {repr(raw)}")
        print(f"  CLEAN : {repr(result['clean_text'])}")
        print(f"  TOKENS ({result['token_count']}):")
        for t in result["tokens"]:
            flags = []
            if t["is_punct"]:         flags.append("PUNCT")
            if t["is_numeric"]:       flags.append("NUM")
            if t["is_emoji"]:         flags.append("EMOJI")
            if t["is_arabic_script"]: flags.append("ARABIC")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            print(f"    pos={t['position']:>2}  raw={repr(t['raw_token']):<25}  clean={repr(t['clean_token']):<20}{flag_str}")
    print(sep)


if __name__ == "__main__":
    run_tests()
