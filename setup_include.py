"""Unpack the INCLUDE dataset zips (Zenodo record 4010759) into ONE folder and show how it is organised.

  python setup_include.py --zips ~/Downloads --list                     # look inside the zips first (extracts nothing)
  python setup_include.py --zips ~/Downloads --out ~/Downloads/include  # extract everything it finds
  python setup_include.py --zips ~/Downloads --out ~/Downloads/include --only Greetings,Pronouns,People
  python setup_include.py --zips ~/Downloads/Greetings_1of2.zip ~/Downloads/Greetings_2of2.zip --out ~/Downloads/include
  python setup_include.py --summary ~/Downloads/include                 # just describe an already unpacked folder

What it does
  * --zips takes folders (every *.zip inside) and/or single zip files. --only keeps the zips whose name starts with
    one of the given words (Greetings, Pronouns, Days_and_Time, People, Home, Places, Jobs, Society ...).
  * checks every zip for corruption (CRC) before extracting, and checks that you have enough free disk space
  * all parts (Greetings_1of2, Greetings_2of2 ...) are merged into the same output folder
  * skips macOS junk (__MACOSX, ._files, .DS_Store) and refuses unsafe paths inside a zip
  * resumable: files that are already there with the right size are skipped, so you can run it again after an interruption
  * prints a summary: categories, number of signs and videos, videos per sign, file types, signs with too few videos,
    and what each label becomes with --auto-labels
  * --delete-zips removes a zip after it extracted without problems (only then)

Next step after it finishes:   python prepare_dataset.py --src <out folder> --max-per-class 80 --auto-labels
"""
import argparse
import shutil
import statistics
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from labels import auto_target

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm", ".m4v"}
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}


def is_junk(name):
    parts = PurePosixPath(name).parts
    if not parts:
        return True
    if "__MACOSX" in parts:
        return True
    base = parts[-1]
    return base.startswith("._") or base.lower() in JUNK_NAMES


def is_unsafe(name):
    """Absolute paths or '..' parts inside a zip (they could write outside the output folder)."""
    p = PurePosixPath(name)
    return p.is_absolute() or ".." in p.parts


ZIP_ERRORS = (zipfile.BadZipFile, zlib.error, EOFError, OSError, RuntimeError, NotImplementedError)


def why(e):
    if isinstance(e, RuntimeError) and "password" in str(e).lower():
        return "the zip is password protected"
    if isinstance(e, (zlib.error, EOFError)):
        return f"damaged or incomplete download ({e}); download this zip again"
    if isinstance(e, zipfile.BadZipFile):
        return f"{e}; the download is incomplete/damaged, or it is a split archive"
    return f"{type(e).__name__}: {e}"


def find_zips(inputs, only):
    zips = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            zips += sorted(p.glob("*.zip"))
        elif p.is_file():
            zips.append(p)
        else:
            print(f"[warn] not found: {p}")
    if only:
        keys = [k.strip().lower() for k in only.split(",") if k.strip()]
        zips = [z for z in zips if any(z.name.lower().startswith(k) for k in keys)]
    seen, out = set(), []
    for z in zips:
        if z.resolve() not in seen:
            seen.add(z.resolve())
            out.append(z)
    return out


def summarize(paths, title, size_bytes=None):
    """paths: relative POSIX paths of FILES. Prints how the dataset is organised."""
    videos = [PurePosixPath(p) for p in paths if PurePosixPath(p).suffix.lower() in VIDEO_EXT]
    others = Counter(PurePosixPath(p).suffix.lower() or "(none)" for p in paths
                     if PurePosixPath(p).suffix.lower() not in VIDEO_EXT)
    print(f"\n=== {title} ===")
    if not videos:
        print("No video files found.", f"Other file types: {dict(others.most_common(6))}" if others else "")
        return
    per_dir = defaultdict(int)
    for v in videos:
        per_dir[v.parent] += 1
    depth = Counter(len(d.parts) for d in per_dir)
    ext = Counter(v.suffix.lower() for v in videos)
    print(f"{len(videos)} videos in {len(per_dir)} folders" + (f", {size_bytes / 1e9:.1f} GB" if size_bytes else ""))
    print("video types:", dict(ext.most_common()))
    print("folder depth holding the videos:", dict(sorted(depth.items())), "(1 = directly under the top folder)")
    if others:
        print("other files (ignored):", dict(others.most_common(5)))

    cats = defaultdict(list)
    for d, n in per_dir.items():
        cats[d.parts[0] if len(d.parts) > 1 else "(top level)"].append((d, n))
    print(f"\n{'category':<22}{'signs':>7}{'videos':>9}{'min':>5}{'median':>8}{'max':>5}")
    for c, items in sorted(cats.items()):
        counts = [n for _, n in items]
        print(f"{c:<22}{len(items):>7}{sum(counts):>9}{min(counts):>5}{int(statistics.median(counts)):>8}{max(counts):>5}")

    print("\nexamples: folder -> label now -> label with --auto-labels")
    for d in sorted(per_dir)[:8]:
        print(f"  {d}  ({per_dir[d]} videos)  ->  '{d.name}'  ->  '{auto_target(d.name)}'")
    names = [d.name for d in per_dir]
    merged = Counter(auto_target(n) for n in names)
    clash = {k: v for k, v in merged.items() if v > 1}
    print(f"\n{len(set(names))} distinct sign folder names; {len(merged)} labels after --auto-labels")
    if clash:
        print("same label from several folders (they will merge):", dict(list(clash.items())[:8]))
    few = sorted((n, d.name) for d, n in per_dir.items() if n < 3)
    if few:
        print(f"[warn] {len(few)} sign folders have fewer than 3 videos, e.g. {few[:4]}")
    if len(depth) > 1:
        print("[note] folders at different depths: fine, the label is always the folder that directly holds the video.")


def list_zip(z):
    with zipfile.ZipFile(z) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir() and not is_junk(i.filename) and not is_unsafe(i.filename)]
    return [i.filename for i in infos], sum(i.file_size for i in infos)


def safe_target(out, name):
    """Join zip member name to out, refusing anything that would land outside out."""
    target = (out / name).resolve()
    try:
        target.relative_to(out.resolve())
    except ValueError:
        return None
    return target


def extract_zip(z, out, verify=True):
    stats = Counter()
    with zipfile.ZipFile(z) as zf:
        if verify:
            bad = zf.testzip()
            if bad:
                raise zipfile.BadZipFile(f"corrupt entry {bad!r}: download this zip again")
        members = [i for i in zf.infolist() if not i.is_dir()]
        total = len(members)
        for k, info in enumerate(members, 1):
            if k % 50 == 0 or k == total:
                print(f"\r  {k}/{total} files", end="", flush=True)
            if is_junk(info.filename):
                stats["junk skipped"] += 1
                continue
            target = None if is_unsafe(info.filename) else safe_target(out, info.filename)
            if target is None:
                print(f"  [warn] unsafe path ignored: {info.filename}")
                stats["unsafe skipped"] += 1
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                stats["already there"] += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                try:
                    with zf.open(info) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    tmp.replace(target)
                finally:
                    tmp.unlink(missing_ok=True)
                stats["extracted"] += 1
    print()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", nargs="+", help="folder(s) with the downloaded zips and/or zip files")
    ap.add_argument("--out", default=None, help="folder to extract into (all parts are merged here)")
    ap.add_argument("--only", default=None, help="comma list of name prefixes, e.g. Greetings,Pronouns,People")
    ap.add_argument("--list", action="store_true", help="only show what is inside the zips (extracts nothing)")
    ap.add_argument("--summary", default=None, metavar="FOLDER", help="describe an already unpacked folder and exit")
    ap.add_argument("--no-verify", action="store_true", help="skip the CRC check (faster, less safe)")
    ap.add_argument("--delete-zips", action="store_true", help="delete each zip after it extracted without problems")
    ap.add_argument("--force", action="store_true", help="continue even if free disk space looks too small")
    args = ap.parse_args()

    if args.summary:
        root = Path(args.summary).expanduser()
        if not root.is_dir():
            raise SystemExit(f"{root} is not a folder")
        files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not is_junk(p.relative_to(root).as_posix())]
        summarize(files, f"folder {root}")
        return
    if not args.zips:
        raise SystemExit("Give --zips <folder or zip files>  (or --summary <folder>). See --help.")
    zips = find_zips(args.zips, args.only)
    if not zips:
        raise SystemExit("No .zip files found (check the path and the --only words).")
    print(f"{len(zips)} zip files:")
    for z in zips:
        print(f"  {z.name}  ({z.stat().st_size / 1e9:.2f} GB)")

    if args.list:
        all_paths, total_size = [], 0
        for z in zips:
            try:
                names, size = list_zip(z)
            except ZIP_ERRORS as e:
                print(f"[error] {z.name}: {why(e)}")
                continue
            print(f"  {z.name}: {len(names)} files, {size / 1e9:.2f} GB unpacked; first entry: {names[0] if names else '-'}")
            all_paths += names
            total_size += size
        summarize(all_paths, "contents of the zips", total_size)
        print("\nNothing was extracted. To extract: add --out <folder> and remove --list.")
        return

    if not args.out:
        raise SystemExit("Give --out <folder to extract into> (or use --list to only look).")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    need = 0
    for z in zips:
        try:
            need += list_zip(z)[1]
        except ZIP_ERRORS:
            pass
    free = shutil.disk_usage(out).free
    print(f"\nneeds about {need / 1e9:.1f} GB, free disk space {free / 1e9:.1f} GB")
    if need * 1.02 > free and not args.force:
        raise SystemExit("Not enough free disk space. Free some space, take fewer categories (--only), or use --force.")

    failed = []
    for z in zips:
        print(f"\n{z.name}")
        try:
            stats = extract_zip(z, out, verify=not args.no_verify)
        except ZIP_ERRORS as e:
            print(f"\n  [error] {why(e)}")
            failed.append(z.name)
            continue
        print("  " + ", ".join(f"{k}: {v}" for k, v in stats.items()))
        if args.delete_zips:
            z.unlink()
            print("  zip deleted")

    files = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and not p.name.endswith(".part")]
    summarize(files, f"unpacked folder {out}")
    if failed:
        print(f"\n[error] these zips failed and must be downloaded again: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nNext:\n  python prepare_dataset.py --src {out} --max-per-class 80 --auto-labels")


if __name__ == "__main__":
    main()
