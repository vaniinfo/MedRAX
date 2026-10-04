"""VinDr protocol, Amendment 2: Kaggle VinBigData release as the data source.

Pre-registered in experiments/vindr_amendment_2_data_source.md. Builds, from the Kaggle
competition "vinbigdata-chest-xray-abnormalities-detection", the files Part A and Part B read
(unchanged scripts):

    python experiments/vindr_kaggle_derive.py sample     # image IDs from the file list -> sample (commit it)
    python experiments/vindr_kaggle_derive.py download   # train.csv + the sampled DICOMs
    python experiments/vindr_kaggle_derive.py derive     # derived labels, boxes and test/ folder

Needs the competition rules accepted on Kaggle and ~/.kaggle/access_token.
"""

import hashlib
import json
import os
import shutil
import zipfile
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SAMPLE = RESULTS / "vindr_kaggle_sample.json"
COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"
RAW = Path("D:/datasets/vindr-kaggle")
DERIVED = Path("D:/datasets/vindr-cxr-kaggle-derived")
SEED = "20261004"
N = 3000
API = "https://www.kaggle.com/api/v1"

CLASSES = {"Cardiomegaly": "Cardiomegaly", "Pleural effusion": "Pleural effusion",
           "Pneumothorax": "Pneumothorax", "Consolidation": "Consolidation", "Atelectasis": "Atelectasis",
           "Edema": "Edema"}  # Kaggle class -> Part A label column ("Edema" only if the class exists)
LATERAL = ["Pleural effusion", "Atelectasis", "Consolidation", "Pneumothorax"]  # Part B classes


def session():
    import requests

    token = (Path.home() / ".kaggle" / "access_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def sample():
    if SAMPLE.exists():
        print(f"{SAMPLE.name} already exists; not rebuilding")
        return
    s, names, token = session(), [], ""
    while True:
        r = s.get(f"{API}/competitions/data/list/{COMPETITION}", params={"pageToken": token, "pageSize": 1000})
        r.raise_for_status()
        body = r.json()
        names += [f.get("name") or f.get("ref") for f in body.get("files", [])]
        token = body.get("nextPageToken") or ""
        if not body.get("hasNextPageToken") or not token:
            break
    ids = sorted({Path(n).stem for n in names if n and n.startswith("train/") and n.endswith(".dicom")})
    source = "competition file list"
    if not ids:
        # Deviation 1 (vindr_amendment_2_deviation_1.md): the file list is empty for this
        # competition, so the IDs come from train.csv's image_id column only; no label column
        # is read here
        import pandas as pd

        download_file(s, "train.csv", RAW / "train.csv")
        ids = sorted(pd.read_csv(RAW / "train.csv", usecols=["image_id"])["image_id"].unique())
        source = "train.csv image_id column only (deviation 1)"
    if not ids:
        raise SystemExit("no training image IDs found")
    rank = lambda i: hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest()
    chosen = sorted(ids, key=rank)[:N]
    RESULTS.mkdir(exist_ok=True)
    SAMPLE.write_text(json.dumps({"competition": COMPETITION, "seed": SEED, "id_source": source,
                                  "listed_train_images": len(ids),
                                  "rule": "lowest SHA-256(seed:image_id), first 3000",
                                  "image_ids": sorted(chosen)}, indent=1), encoding="utf-8")
    print(f"{len(ids)} training images listed, {len(chosen)} sampled -> {SAMPLE}; commit it before downloading")


def download_file(s, name, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with s.get(f"{API}/competitions/data/download/{COMPETITION}/{name}", stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    # Kaggle sends some files zipped: unpack the single member under the expected name
    if zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as z:
            members = z.namelist()
            assert len(members) == 1, f"{name}: expected one file in the zip, found {members}"
            with z.open(members[0]) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
        tmp.unlink()
    else:
        tmp.replace(dst)


def download():
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["image_ids"]
    s = session()
    download_file(s, "train.csv", RAW / "train.csv")
    for n, i in enumerate(ids, 1):
        download_file(s, f"train/{i}.dicom", RAW / "train" / f"{i}.dicom")
        if n % 100 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} DICOMs", flush=True)


def sides_of(row):
    """Sides a box marks (percent of width): both if it crosses the midline region."""
    if row["x1"] < 40 and row["x2"] > 60:
        return {"right", "left"}
    return {"right" if (row["x1"] + row["x2"]) / 2 < 50 else "left"}


def derive():
    import pandas as pd
    import pydicom

    if (DERIVED / "annotations" / "image_labels_test.csv").exists():
        print("derived files already exist; not rebuilding")
        return
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["image_ids"]
    t = pd.read_csv(RAW / "train.csv")
    for col in ("image_id", "class_name", "rad_id", "x_min", "y_min", "x_max", "y_max"):
        assert col in t.columns, f"column {col} missing in train.csv; columns are {list(t.columns)}"
    t = t[t["image_id"].isin(ids)]
    readers = t.groupby("image_id")["rad_id"].nunique()
    excluded = sorted(set(ids) - set(readers[readers == 3].index))
    keep = [i for i in ids if i not in excluded and (RAW / "train" / f"{i}.dicom").exists()]
    missing = sorted(set(ids) - set(excluded) - set(keep))

    labels, boxes, single, conflict = [], [], {}, {}
    for image_id in keep:
        rows = t[t["image_id"] == image_id]
        ds = pydicom.dcmread(str(RAW / "train" / f"{image_id}.dicom"), stop_before_pixels=True)
        width = int(ds.Columns)
        label = {"image_id": image_id}
        for cls, column in CLASSES.items():
            marked = rows[rows["class_name"] == cls]
            n_readers = marked["rad_id"].nunique()
            label[column] = int(n_readers >= 2)
            if n_readers == 1:
                single[cls] = single.get(cls, 0) + 1
            if cls not in LATERAL or n_readers < 2:
                continue
            marked = marked.assign(x1=marked["x_min"] / width * 100, x2=marked["x_max"] / width * 100)
            by_side = {s: {r.rad_id for r in marked.itertuples() if s in sides_of({"x1": r.x1, "x2": r.x2})}
                       for s in ("right", "left")}
            confirmed = [s for s, rs in by_side.items() if len(rs) >= 2]
            if not confirmed:
                conflict[cls] = conflict.get(cls, 0) + 1
                continue
            for r in marked.itertuples():
                s = sides_of({"x1": r.x1, "x2": r.x2})
                if len(confirmed) == 2 or (s == {confirmed[0]}):
                    boxes.append({"image_id": image_id, "class_name": cls, "x_min": r.x_min,
                                  "y_min": r.y_min, "x_max": r.x_max, "y_max": r.y_max})
        labels.append(label)

    (DERIVED / "annotations").mkdir(parents=True, exist_ok=True)
    (DERIVED / "test").mkdir(exist_ok=True)
    pd.DataFrame(labels).to_csv(DERIVED / "annotations" / "image_labels_test.csv", index=False)
    pd.DataFrame(boxes, columns=["image_id", "class_name", "x_min", "y_min", "x_max", "y_max"]).to_csv(
        DERIVED / "annotations" / "annotations_test.csv", index=False)
    for image_id in keep:
        src, dst = RAW / "train" / f"{image_id}.dicom", DERIVED / "test" / f"{image_id}.dicom"
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    report = {"sampled": len(ids), "used": len(keep), "excluded_not_three_readers": excluded,
              "missing_dicom": missing, "marked_by_exactly_one_reader": single,
              "part_b_side_conflicts_excluded": conflict,
              "edema_class_in_train_csv": bool((t["class_name"] == "Edema").any())}
    (DERIVED / "derivation_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    {"sample": sample, "download": download, "derive": derive}[sys.argv[1]]()
