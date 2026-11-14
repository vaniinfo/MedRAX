"""Downloads only the listed CheXpert Plus PNGs (logs/chexpert_plus_candidates.files.txt) from Redivis, keeping
their folder names, into D:/datasets/chexpert_plus/images/. Run it yourself: the first run asks you to log in to
Redivis in the browser (your own account; nothing is stored in this repository). Files already on disk are skipped.

    .venv/Scripts/python.exe experiments/chexpert_plus_download.py TABLE_REFERENCE

TABLE_REFERENCE: the reference of the PNG_train file table, copied from Redivis (Files tab > PNG_train folder >
menu > copy reference), e.g. "stanfordaimi.chexpert_plus:xxxx.png_train:yyyy".
"""

import sys
from pathlib import Path

import redivis

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "logs" / "chexpert_plus_candidates.files.txt"
DEST = Path("D:/datasets/chexpert_plus/images")


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    table = redivis.table(sys.argv[1])
    paths = [p.strip() for p in LIST.read_text(encoding="utf-8").splitlines() if p.strip()]
    ok = failed = 0
    for p in paths:  # e.g. PNG_train/patient01673/study5/view1_frontal.png
        target = DEST / p
        if target.exists():
            ok += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        inside = p.split("/", 1)[1]  # the path inside the PNG_train table
        try:
            f = table.file(inside)
            f.download(str(target.parent))  # into the existing folder, under the file's own name
            ok += 1
            print(f"ok   {p}", flush=True)
        except Exception as e:  # report and go on with the rest
            failed += 1
            print(f"FAIL {p}: {e}", flush=True)
    print(f"{ok} of {len(paths)} files in {DEST}; {failed} failed")


if __name__ == "__main__":
    main()
