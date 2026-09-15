"""Download a larger labelled evaluation set from NIH Open-i (Indiana University CXRs).

Same collection as the Kaggle raddar/chest-xrays-indiana-university dataset, but the
API gives us the curated MeSH labels alongside each image, which is what we need for
ground truth. Filename prefixes are search queries, NOT labels -- Open-i matches
negated text too. Truth always comes from the MeSH `problems` field.
"""
import csv, json, os, time, urllib.parse, urllib.request

BASE = "https://openi.nlm.nih.gov"
OUT = "/Users/hari/_workarea_/vani/MedRAX/data/indiana_eval"
os.makedirs(OUT, exist_ok=True)

QUERIES = [
    ("normal", "normal", 70),
    ("cardiomegaly", "cardiomegaly", 45),
    ("effusion", "pleural effusion", 45),
    ("pneumothorax", "pneumothorax", 40),
    ("edema", "pulmonary edema", 40),
    ("atelectasis", "atelectasis", 40),
    ("opacity", "lung opacity", 35),
    ("nodule", "pulmonary nodule", 35),
    ("consolidation", "consolidation", 30),
]

def get(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()

rows, seen = [], set()
for label, query, want in QUERIES:
    url = (f"{BASE}/api/search?query={urllib.parse.quote(query)}"
           f"&coll=cxr&m=1&n={want * 2}&favor=r")
    try:
        data = json.loads(get(url))
    except Exception as exc:
        print(f"  ! query {query!r} failed: {exc}", flush=True)
        continue
    got = 0
    for rec in data.get("list", []):
        if got >= want:
            break
        uid, img = rec.get("uid"), rec.get("imgLarge")
        if not uid or not img or uid in seen:
            continue
        fname = f"{label}_{uid}.png"
        try:
            blob = get(BASE + img)
        except Exception:
            continue
        open(os.path.join(OUT, fname), "wb").write(blob)
        seen.add(uid); got += 1
        mesh = rec.get("MeSH") or {}
        rows.append({"file": fname, "uid": uid, "query": query,
                     "problems": rec.get("Problems") or "",
                     "mesh_major": "; ".join(mesh.get("major") or []),
                     "mesh_minor": "; ".join(mesh.get("minor") or []),
                     "impression": (rec.get("impression") or "").strip().replace("\n", " ")})
        time.sleep(0.15)
    print(f"{label:15s} {got:3d} images", flush=True)

with open(os.path.join(OUT, "GROUND_TRUTH.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} unique images -> {OUT}", flush=True)
