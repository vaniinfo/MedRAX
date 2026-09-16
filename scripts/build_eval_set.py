"""Build the evaluation set from the Kaggle mirror of the Indiana University CXRs.

Replaces fetch_openi_eval_set.py for this path. Open-i serves the same collection
through a search API that throttles to roughly two queries per cooldown window and
reports the throttle as HTTP 200 carrying an HTML error page -- which is how four of
those pages came to sit in data/indiana_eval named .png. The Kaggle mirror is the same
images and the same curated MeSH labels, with no throttle and no ranked-search
sampling, so the set is reproducible from a seed rather than redrawn each time.

Selection is stratified: up to --cap positives for each finding, plus a block of
confirmed-normal negatives. Enriching positives is sound here because both statistics
computed downstream -- AUC, and balanced accuracy at the chosen threshold -- are
prevalence-independent. Raising the positive rate buys precision on the rare findings
without biasing either number.

One corpus ceiling worth knowing before reading any pneumothorax row: there are 28
pneumothorax positives in the entire 3818-image frontal collection. That is not a
sampling limit that a bigger download would relieve. It is all there is.

Prerequisites: scripts/findings.py, and the two label CSVs in data/indiana_kaggle:
    python -m kaggle datasets download -d raddar/chest-xrays-indiana-university \\
        -f indiana_reports.csv -p data/indiana_kaggle --unzip

  python scripts/build_eval_set.py --dry-run     # show the selection, download nothing
  python scripts/build_eval_set.py               # build it, ~10 min
"""
import argparse
import csv
import os
import random
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from findings import FINDINGS, MIN_POSITIVES, is_positive

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "indiana_kaggle")
OUT = os.path.join(ROOT, "data", "indiana_eval")
GROUND_TRUTH = os.path.join(OUT, "GROUND_TRUTH.csv")

DATASET = "raddar/chest-xrays-indiana-university"
REMOTE_DIR = "images/images_normalized"
FIELDS = ["file", "uid", "selected_for", "problems", "mesh", "impression"]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def is_png(path):
    """Validate content, not merely presence.

    The Open-i fetch trusted that a file existing meant a file downloaded, and left
    four HTML error pages on disk with .png names. The lesson outlives that script.
    """
    try:
        with open(path, "rb") as f:
            return f.read(8) == PNG_MAGIC
    except OSError:
        return False


def load_frontal(exclude=frozenset()):
    """Join reports to projections on uid, keeping frontal views.

    Laterals are dropped: every tool in this pipeline is trained on frontal films, and
    a lateral scored against a report-derived label measures the wrong thing.
    """
    reports_csv = os.path.join(RAW, "indiana_reports.csv")
    proj_csv = os.path.join(RAW, "indiana_projections.csv")
    for path in (reports_csv, proj_csv):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\nSee the prerequisites in this script's docstring.")

    with open(reports_csv, encoding="utf-8") as f:
        reports = {r["uid"]: r for r in csv.DictReader(f)}
    with open(proj_csv, encoding="utf-8") as f:
        projections = list(csv.DictReader(f))

    rows = []
    for entry in projections:
        report = reports.get(entry["uid"])
        if not report or entry["projection"] != "Frontal":
            continue
        # Held-out draws must not reuse a film the tables were fitted on, or the
        # experiment measures memory rather than generalisation.
        if entry["filename"] in exclude:
            continue
        rows.append({"file": entry["filename"], "uid": entry["uid"],
                     "problems": report["Problems"] or "",
                     "mesh": report["MeSH"] or "",
                     "impression": (report["impression"] or "").strip().replace("\n", " ")})
    return rows


def select(rows, cap, normals, seed):
    """Stratified pick: up to `cap` positives per finding, plus `normals` negatives.

    Sorted before shuffling so the result depends on the seed alone and not on the
    order the CSV happened to be in.
    """
    rng = random.Random(seed)
    picked = {}

    def take(row, reason):
        picked.setdefault(row["file"], dict(row, selected_for=set()))
        picked[row["file"]]["selected_for"].add(reason)

    for finding in FINDINGS:
        pool = sorted((r for r in rows if is_positive(r["problems"], finding)),
                      key=lambda r: r["file"])
        rng.shuffle(pool)
        # An image already picked for another finding still counts against this
        # finding's cap -- it genuinely is one of its positives.
        for row in pool[:cap]:
            take(row, finding)

    pool = sorted((r for r in rows if r["problems"].strip().lower() == "normal"),
                  key=lambda r: r["file"])
    rng.shuffle(pool)
    for row in pool[:normals]:
        take(row, "normal")

    return list(picked.values())


def download(selected):
    """Fetch the selected images, skipping any already present and valid."""
    import kaggle

    todo = [r for r in selected if not is_png(os.path.join(OUT, r["file"]))]
    print(f"\nimages: {len(selected) - len(todo)} already valid, {len(todo)} to download")
    if not todo:
        return []

    failed, t0 = [], time.time()
    for n, row in enumerate(todo, 1):
        final = os.path.join(OUT, row["file"])
        try:
            kaggle.api.dataset_download_file(
                DATASET, f"{REMOTE_DIR}/{row['file']}", path=OUT, force=True)
            # The API delivers one zip per file rather than the bare image.
            archive = final + ".zip"
            if os.path.exists(archive):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(OUT)
                os.remove(archive)
            if not is_png(final):
                raise RuntimeError("downloaded file is not a PNG")
        except Exception as exc:
            print(f"  ! {row['file']}: {exc}", flush=True)
            failed.append(row["file"])
            if os.path.exists(final):
                os.remove(final)
        if n % 25 == 0:
            rate = (time.time() - t0) / n
            print(f"  {n}/{len(todo)}  (~{rate * (len(todo) - n):.0f}s left)", flush=True)
    return failed


def report_balance(rows, label):
    print(f"\nclass balance -- {label} ({len(rows)} images):")
    print(f"  {'finding':18s} {'n+':>5s} {'n-':>5s} {'prevalence':>11s}")
    thin = []
    for finding in FINDINGS:
        pos = sum(1 for r in rows if is_positive(r["problems"], finding))
        flag = ""
        if pos < MIN_POSITIVES:
            flag = f"   <-- under {MIN_POSITIVES}"
            thin.append(finding)
        print(f"  {finding:18s} {pos:5d} {len(rows) - pos:5d} "
              f"{pos / len(rows):10.1%}{flag}")
    return thin


def main():
    global OUT, GROUND_TRUTH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap", type=int, default=100,
                    help="max positives per finding (default 100)")
    ap.add_argument("--normals", type=int, default=150,
                    help="confirmed-normal negatives to add (default 150)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection without downloading anything")
    ap.add_argument("--out", default=OUT, help="destination directory")
    ap.add_argument("--exclude", default="",
                    help="a GROUND_TRUTH.csv whose films must NOT be drawn again")
    args = ap.parse_args()

    OUT = args.out
    GROUND_TRUTH = os.path.join(OUT, "GROUND_TRUTH.csv")

    exclude = set()
    if args.exclude:
        with open(args.exclude, newline="", encoding="utf-8") as handle:
            exclude = {r["file"] for r in csv.DictReader(handle)}
        print(f"excluding {len(exclude)} films already used for fitting")

    rows = load_frontal(exclude)
    print(f"{len(rows)} frontal images available with a matching report")
    report_balance(rows, "full collection")

    selected = select(rows, args.cap, args.normals, args.seed)
    thin = report_balance(selected, f"selection (cap={args.cap}, seed={args.seed})")
    size_gb = len(selected) * 2.0 / 1024
    print(f"\n{len(selected)} images selected, roughly {size_gb:.1f} GB to download")
    if thin:
        print(f"still under {MIN_POSITIVES} positives: {', '.join(thin)}")
        print("analyze_reliability.py will leave those out of RELIABILITY.")

    if args.dry_run:
        print("\n--dry-run: nothing downloaded")
        return 0

    os.makedirs(OUT, exist_ok=True)
    failed = download(selected)
    have = [r for r in selected if is_png(os.path.join(OUT, r["file"]))]
    if failed:
        print(f"\n!! {len(failed)} download(s) failed; re-run to retry just those")

    with open(GROUND_TRUTH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in have:
            writer.writerow(dict(row, selected_for=";".join(sorted(row["selected_for"]))))
    print(f"\n{len(have)} images -> {os.path.relpath(GROUND_TRUTH)}")
    report_balance(have, "written")
    print("\nNext: python scripts/measure_reliability.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
