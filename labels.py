"""Label cleanup / merging helpers (pure Python, no heavy imports).

Used by merge_labels.py (fix data you already converted) and prepare_dataset.py (fix labels while converting).

A label-map file is JSON, either of these shapes (or a mix):
    {"E": ["E1", "E2"], "9": ["9a", "9b"]}        target -> list of variants
    {"E1": "E", "E2": "E", "9a": "9"}             variant -> target
"""
import json
import re
from pathlib import Path


def load_map(path):
    """Return {source_label: target_label}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            for s in v:
                out[str(s)] = str(k)
        else:
            out[str(k)] = str(v)
    return out


def auto_target(name):
    """Conservative automatic clean-up of variant names. Always review the dry-run plan.

        '12. Hello' -> 'Hello'     'hello (2)' -> 'hello'     'hello_2' -> 'hello'
        'E1' -> 'E'                '9a' -> '9'                'A_2' -> 'A'
    Plain numbers ('10', '21') and ordinary words are left alone.
    """
    s = name.strip()
    s = re.sub(r"^\d+\s*[\.\-_)]\s*", "", s)           # numbering prefix
    s = re.sub(r"\s*[\(\[]\d+[\)\]]$", "", s)          # '(2)' suffix
    if len(s) > 2:
        s = re.sub(r"[_\-\s]+(?:v|var|variant)?\d{1,2}$", "", s)   # '_2' suffix
    if re.fullmatch(r"[A-Za-z]\d", s):                 # E1 -> E
        s = s[0]
    elif re.fullmatch(r"\d[a-z]", s):                  # 9a -> 9
        s = s[0]
    return s or name


def resolve(name, mapping=None, auto=False, merge_case=False):
    """Final label for one source label."""
    mapping = mapping or {}
    if name in mapping:
        t = mapping[name]
    else:
        t = auto_target(name) if auto else name
        t = mapping.get(t, t)
    if merge_case:
        t = t.upper() if len(t) == 1 else t.lower()
    return t


def plan_labels(names, mapping=None, auto=False, merge_case=False):
    """{target: sorted source names} for every target that actually changes something
    (a rename, or several sources folding into one)."""
    groups = {}
    for n in names:
        groups.setdefault(resolve(n, mapping, auto, merge_case), []).append(n)
    return {t: sorted(s) for t, s in groups.items() if s != [t]}
