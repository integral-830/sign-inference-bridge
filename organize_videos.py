"""Sort a video dataset whose labels are in a CSV / JSON file into one folder per sign.

Many datasets (ASL Citizen, CISLR, WLASL, MS-ASL ...) ship ONE big folder of videos plus a label file.
prepare_dataset.py needs  <folder>/<sign>/<video>, so run this first:

  python organize_videos.py --videos ~/Downloads/cislr/videos --csv ~/Downloads/cislr/dataset.csv --list-columns
  python organize_videos.py --videos ~/Downloads/cislr/videos --csv ~/Downloads/cislr/dataset.csv --out ~/Downloads/cislr_sorted
  python organize_videos.py --videos ~/Downloads/wlasl/videos --wlasl-json ~/Downloads/wlasl/WLASL_v0.3.json --top-classes 100
  python prepare_dataset.py --src ~/Downloads/cislr_sorted --max-per-class 60

Columns are guessed (file: video/path/file..., label: gloss/label/word/sign/text...). If the guess is wrong,
run --list-columns and pass --file-col / --label-col. A split column can filter (--split-col split --splits train,val).
By default files are symlinked (no extra disk space); use --mode copy if you plan to move the folder.
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}
FILE_GUESS = ["videofile", "video", "videopath", "path", "file", "filename", "videoid", "vid", "uid", "id",
              "clip", "clipname", "name"]
LABEL_GUESS = ["gloss", "label", "word", "sign", "signname", "text", "class", "category", "annotation", "lemma",
               "translation"]


def norm(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def safe(name):
    return re.sub(r"[^\w\-. ]+", "_", str(name)).strip()


def read_csv(path, sep=None):
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        if sep is None:
            try:
                sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
        rd = csv.DictReader(f, delimiter=sep)
        return rd.fieldnames or [], list(rd)


def pick_col(fields, wanted, guesses, what):
    by_norm = {norm(c): c for c in fields}
    if wanted:
        if norm(wanted) in by_norm:
            return by_norm[norm(wanted)]
        raise SystemExit(f"--{what}-col '{wanted}' not found. Columns: {fields}")
    for g in guesses:
        if g in by_norm:
            return by_norm[g]
    raise SystemExit(f"Could not guess the {what} column. Columns: {fields}\nPass --{what}-col NAME.")


def index_videos(root):
    by_name, by_stem = {}, {}
    for p in sorted(Path(root).expanduser().rglob("*")):
        if p.is_file() and p.suffix.lower() in VID_EXT and not p.name.startswith("."):
            by_name.setdefault(p.name.lower(), p)
            by_stem.setdefault(p.stem.lower(), p)
    return by_name, by_stem


def find_video(value, by_name, by_stem):
    base = os.path.basename(str(value).strip().replace("\\", "/"))
    if not base:
        return None
    return by_name.get(base.lower()) or by_stem.get(Path(base).stem.lower()) or by_stem.get(base.lower())


def place(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            mode = "copy"
    (shutil.move if mode == "move" else shutil.copy2)(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder containing the videos (searched recursively)")
    ap.add_argument("--csv", default=None, help="label file (csv/tsv)")
    ap.add_argument("--wlasl-json", default=None, help="WLASL_v0.3.json style file")
    ap.add_argument("--out", default="organized_videos")
    ap.add_argument("--file-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--sep", default=None, help="csv delimiter (auto-detected by default)")
    ap.add_argument("--split-col", default=None)
    ap.add_argument("--splits", default=None, help="comma list to keep, e.g. train,val")
    ap.add_argument("--mode", choices=["symlink", "copy", "move"], default="symlink")
    ap.add_argument("--min-per-class", type=int, default=1)
    ap.add_argument("--top-classes", type=int, default=0, help="keep only the N signs with most videos")
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--list-columns", action="store_true", help="print the columns and first rows, then exit")
    args = ap.parse_args()

    if not args.csv and not args.wlasl_json:
        raise SystemExit("Give --csv FILE or --wlasl-json FILE")
    keep_splits = {s.strip().lower() for s in args.splits.split(",")} if args.splits else None

    rows = []  # (file_value, label)
    if args.wlasl_json:
        for entry in json.loads(Path(args.wlasl_json).expanduser().read_text(encoding="utf-8")):
            for inst in entry.get("instances", []):
                if keep_splits and str(inst.get("split", "")).lower() not in keep_splits:
                    continue
                rows.append((str(inst.get("video_id", "")), entry.get("gloss", "")))
    else:
        fields, data = read_csv(Path(args.csv).expanduser(), args.sep)
        if args.list_columns:
            print("Columns:", fields)
            for r in data[:3]:
                print("  ", dict(list(r.items())[:8]))
            print(f"{len(data)} rows. Pass --file-col / --label-col if the guess below is wrong.")
            try:
                print(f"Guess: file column = {pick_col(fields, args.file_col, FILE_GUESS, 'file')!r}, "
                      f"label column = {pick_col(fields, args.label_col, LABEL_GUESS, 'label')!r}")
            except SystemExit as e:
                print(e)
            return
        fcol = pick_col(fields, args.file_col, FILE_GUESS, "file")
        lcol = pick_col(fields, args.label_col, LABEL_GUESS, "label")
        scol = pick_col(fields, args.split_col, [], "split") if args.split_col else None
        for r in data:
            if keep_splits and scol and str(r.get(scol, "")).strip().lower() not in keep_splits:
                continue
            rows.append((r.get(fcol, ""), r.get(lcol, "")))

    by_name, by_stem = index_videos(args.videos)
    print(f"{len(by_name)} video files found, {len(rows)} label rows")
    groups, missing = defaultdict(list), []
    seen = set()
    for value, label in rows:
        label = safe(label)
        if not label:
            continue
        p = find_video(value, by_name, by_stem)
        if p is None:
            missing.append(value)
        elif p not in seen:
            seen.add(p)
            groups[label].append(p)

    groups = {k: v for k, v in groups.items() if len(v) >= args.min_per_class}
    if args.top_classes:
        top = sorted(groups, key=lambda k: (-len(groups[k]), k))[: args.top_classes]
        groups = {k: groups[k] for k in top}
    out = Path(args.out).expanduser()
    n = 0
    for label, files in sorted(groups.items()):
        files = sorted(files)
        if args.max_per_class:
            random.Random(0).shuffle(files)
            files = sorted(files[: args.max_per_class])
        for i, p in enumerate(files):
            place(p, out / label / f"{i:04d}_{p.name}", args.mode)
            n += 1
    print(f"Sorted {n} videos into {len(groups)} sign folders under {out}")
    if missing:
        print(f"{len(missing)} rows had no matching video file (e.g. {missing[:3]})")
    top10 = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]
    print("Largest signs:", ", ".join(f"{k} ({len(v)})" for k, v in top10))
    print(f"\nNext: python prepare_dataset.py --src {out} --max-per-class 60")


if __name__ == "__main__":
    main()
