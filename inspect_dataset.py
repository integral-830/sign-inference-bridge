"""Look at a downloaded dataset BEFORE converting it:  python inspect_dataset.py ~/Downloads/include

Shows which folders directly contain images/videos (those folder names become the labels), how many files each
has, and what --auto-labels would turn the names into. Nothing is changed.
"""
import sys
from collections import Counter
from pathlib import Path

from labels import auto_target

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1]).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a folder")
    per_dir, other = {}, Counter()
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            ext = p.suffix.lower()
            if ext in IMG | VID:
                per_dir.setdefault(p.parent, Counter())["img" if ext in IMG else "vid"] += 1
            else:
                other[ext or "(none)"] += 1
    if not per_dir:
        raise SystemExit("No images or videos found. Did you unzip? Is this the right folder?")
    total_i = sum(c["img"] for c in per_dir.values())
    total_v = sum(c["vid"] for c in per_dir.values())
    print(f"{len(per_dir)} folders hold media: {total_i} images, {total_v} videos")
    depths = Counter(len(d.relative_to(root).parts) for d in per_dir)
    print("depth of those folders below the root:", dict(sorted(depths.items())))
    if len(depths) > 1:
        print("  NOTE: mixed depths. That is fine (the label is always the immediate parent folder).")
    counts = sorted(((sum(c.values()), d) for d, c in per_dir.items()))
    print(f"files per folder: min {counts[0][0]}, max {counts[-1][0]}")
    print("\nfirst 15 folders -> label now -> label with --auto-labels")
    for d in sorted(per_dir)[:15]:
        n = sum(per_dir[d].values())
        print(f"  {d.relative_to(root)}  ({n} files)  ->  '{d.name}'  ->  '{auto_target(d.name)}'")
    names = [d.name for d in per_dir]
    merged = Counter(auto_target(n) for n in names)
    clash = {k: v for k, v in merged.items() if v > 1}
    print(f"\n{len(set(names))} distinct label names; {len(merged)} after --auto-labels")
    if clash:
        print("these would merge (same sign split over several folders):", dict(list(clash.items())[:10]))
    small = [d.name for n, d in counts if n < 3]
    if small:
        print(f"WARNING: {len(small)} folders have fewer than 3 files, e.g. {small[:5]}")
    if other:
        print("other file types ignored:", dict(other.most_common(5)))
    print("\nIf the labels above look like sign names, run prepare_dataset.py --src <this folder> ...")


if __name__ == "__main__":
    main()
