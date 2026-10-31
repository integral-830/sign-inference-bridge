"""Merge / rename sign folders inside data/sequences so variants become ONE sign.

Typical use: the ISL static-gesture dataset has two versions of E (E1, E2) and of 9 (9a, 9b).
Merge them, otherwise 'E1' is not a single letter and will not join into spelled words.

  python merge_labels.py --auto                      # DRY RUN: shows what would be merged
  python merge_labels.py --auto --apply              # do it
  python merge_labels.py --map labelmap.json --apply # explicit mapping (see labelmap_example.json)
  python merge_labels.py --map labelmap.json --auto --merge-case --apply

--auto        clean obvious variants: E1/E2 -> E, 9a/9b -> 9, 'hello_2' -> 'hello', '12. Hello' -> 'Hello'
--merge-case  treat 'Hello' and 'hello' as one sign (single characters become upper case)
Nothing is changed unless you pass --apply. Files are moved, never deleted.
"""
import argparse
import shutil
from pathlib import Path

from labels import load_map, plan_labels


def merge_into(root, src_name, dst_name):
    """Move every .npy from root/src_name into root/dst_name (works on case-insensitive disks too)."""
    src, dst = root / src_name, root / dst_name
    if not src.exists():
        return 0
    if src_name.lower() == dst_name.lower():  # case-only rename: go through a temp name
        tmp = root / f".{src_name}.__tmp"
        src.rename(tmp)
        src = tmp
    dst.mkdir(exist_ok=True)
    moved = 0
    for f in sorted(src.glob("*.npy")):
        target = dst / f.name
        if target.exists():
            target = dst / f"{src_name}__{f.name}"
        shutil.move(str(f), str(target))
        moved += 1
    try:
        src.rmdir()
    except OSError:
        pass  # something else is still inside; leave it
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sequences")
    ap.add_argument("--map", default=None, help="JSON label map (see labelmap_example.json)")
    ap.add_argument("--auto", action="store_true", help="also merge obvious variants (E1/E2 -> E ...)")
    ap.add_argument("--merge-case", action="store_true", help="'Hello' == 'hello'; single characters upper case")
    ap.add_argument("--apply", action="store_true", help="actually move files (default is a dry run)")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f"{root} not found")
    names = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    mapping = load_map(args.map) if args.map else None
    plan = plan_labels(names, mapping, args.auto, args.merge_case)
    if not plan:
        print("Nothing to merge or rename.")
        return
    count = lambda n: len(list((root / n).glob("*.npy")))
    print(f"{'target':<20} <- sources (samples)")
    for t, srcs in sorted(plan.items()):
        print(f"{t:<20} <- " + ", ".join(f"{s} ({count(s)})" for s in srcs))
    if not args.apply:
        print("\nDry run only. Re-run with --apply to do this.")
        return
    total = 0
    for t, srcs in plan.items():
        for s in srcs:
            if s != t:
                total += merge_into(root, s, t)
    print(f"\nDone: moved {total} files.")


if __name__ == "__main__":
    main()
