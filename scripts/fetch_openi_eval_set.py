"""Download a labelled evaluation set from NIH Open-i (Indiana University CXRs).

Same collection as the Kaggle raddar/chest-xrays-indiana-university dataset, but the
API gives us the curated MeSH labels alongside each image, which is what we need for
ground truth. Filename prefixes are search queries, NOT labels -- Open-i matches
negated text too. Truth always comes from the MeSH `problems` field.

Two phases, because the labels are the irreplaceable half of this:

  1. metadata   one cheap API call per query -> MANIFEST.csv. Nothing downloaded yet.
  2. images     one download per manifest row. Resumable, with backoff.

The previous single-phase version downloaded as it went, and Open-i throttled it after
about 175 images. It then wrote GROUND_TRUTH.csv from the four queries that had
completed, with nothing recorded to say the other five had failed -- a truncated set
that looked complete. GROUND_TRUTH.csv is now written only when the manifest is whole
and every image is on disk, unless --allow-partial says to accept less.

  python scripts/fetch_openi_eval_set.py              # fetch, or resume a fetch
  python scripts/fetch_openi_eval_set.py --refresh    # re-query Open-i from scratch
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from findings import FINDINGS, MIN_POSITIVES, is_positive

BASE = "https://openi.nlm.nih.gov"
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "indiana_eval")
MANIFEST = os.path.join(OUT, "MANIFEST.csv")
GROUND_TRUTH = os.path.join(OUT, "GROUND_TRUTH.csv")
# Per-query results, banked as each one succeeds. The throttle is aggressive enough
# that nine queries may not fit in a single run, and without this a failure on the
# last one would discard the eight that had already worked.
QUERY_CACHE = os.path.join(OUT, "QUERY_CACHE.json")

# Quotas are sized by measured yield, not by eye. Open-i matches negated text, so a
# query returns a mix of positives and ruled-out mentions, and the ratio is nothing
# like uniform across findings. Measured on the 175-image partial set:
#
#   cardiomegaly      45/45  100%   ranked search returns well-indexed positives
#   pleural effusion  45/45  100%
#   pneumothorax      13/35   37%   mostly reports saying "no pneumothorax"
#
# A quota of 40 on pneumothorax therefore yields ~15 positives and can never clear
# MIN_POSITIVES, which is why that row kept coming back UNMEASURED. Findings whose
# yield is unknown (consolidation, edema -- their queries have never run) are sized as
# if they behave like pneumothorax rather than like cardiomegaly. Over-fetching costs
# a few minutes; under-fetching costs a whole measurement run.
QUERIES = [
    ("normal", "normal", 70),
    ("cardiomegaly", "cardiomegaly", 45),
    ("effusion", "pleural effusion", 45),
    ("pneumothorax", "pneumothorax", 90),
    ("edema", "pulmonary edema", 70),
    ("atelectasis", "atelectasis", 40),
    ("opacity", "lung opacity", 35),
    ("nodule", "pulmonary nodule", 35),
    ("consolidation", "consolidation", 70),
]

# `img` is the download path, kept in the manifest so phase 2 needs no further queries.
# GROUND_TRUTH.csv keeps the original columns, since measure_reliability.py reads it.
MANIFEST_FIELDS = ["file", "uid", "query", "img", "problems",
                   "mesh_major", "mesh_minor", "impression"]
GROUND_TRUTH_FIELDS = ["file", "uid", "query", "problems",
                       "mesh_major", "mesh_minor", "impression"]


# Seconds between /api/search calls, and between image downloads. The search endpoint
# is throttled far harder than the image endpoint -- the original run managed 175
# consecutive image downloads but died on its fifth query.
QUERY_SPACING = 60.0
IMAGE_SPACING = 0.5

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def looks_like_image(blob):
    return blob.startswith(PNG_MAGIC) or blob.startswith(JPEG_MAGIC)


def get(url, timeout=45, tries=5, want="json", backoff=30.0):
    """GET with exponential backoff, validating that the answer is what was asked for.

    Open-i reports throttling as **HTTP 200 carrying an HTML error page**, not as 429
    or 503. Nothing raises, so a version that only retried socket exceptions could not
    see it at all, and it cost this script twice:

      * eight of nine metadata queries were abandoned on the first throttle, because
        the HTML only failed later, at json.loads, outside the retry;
      * four error pages were written into data/indiana_eval as .png files and counted
        as successful downloads -- 13854 bytes of HTML each, all pneumothorax.

    So the check belongs here, where a retry can still act on it.
    """
    delay = backoff
    for attempt in range(1, tries + 1):
        reason = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                ctype = (response.headers.get("Content-Type") or "").lower()
                blob = response.read()
            if want == "json":
                if "json" in ctype:
                    return json.loads(blob)
                reason = f"throttled? expected JSON, got {ctype.split(';')[0] or 'nothing'}"
            else:
                if looks_like_image(blob):
                    return blob
                reason = f"throttled? expected an image, got {ctype.split(';')[0]}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if attempt == tries:
            raise RuntimeError(reason)
        print(f"      retry {attempt}/{tries - 1} in {delay:.0f}s ({reason})", flush=True)
        time.sleep(delay)
        delay *= 2


def collect_metadata():
    """One API call per query, no image downloads. Returns (rows, failed_queries).

    Each query's results are banked to QUERY_CACHE.json the moment they arrive, so a
    re-run after a throttle resumes at the first query that has not succeeded yet
    rather than spending the request budget re-asking questions already answered.
    """
    cache = {}
    if os.path.exists(QUERY_CACHE):
        with open(QUERY_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        if cache:
            print(f"  {len(cache)} of {len(QUERIES)} queries already banked from an "
                  "earlier run", flush=True)

    rows, seen, failed = [], set(), []
    for label, query, want in QUERIES:
        if label not in cache:
            url = (f"{BASE}/api/search?query={urllib.parse.quote(query)}"
                   f"&coll=cxr&m=1&n={want * 2}&favor=r")
            try:
                data = get(url, want="json")
            except Exception as exc:
                print(f"  ! query {query!r} failed: {exc}", flush=True)
                failed.append(query)
                continue
            fresh, taken = [], set()
            for rec in data.get("list", []):
                if len(fresh) >= want:
                    break
                uid, img = rec.get("uid"), rec.get("imgLarge")
                if not uid or not img or uid in taken:
                    continue
                mesh = rec.get("MeSH") or {}
                fresh.append({
                    "file": f"{label}_{uid}.png", "uid": uid, "query": query, "img": img,
                    "problems": rec.get("Problems") or "",
                    "mesh_major": "; ".join(mesh.get("major") or []),
                    "mesh_minor": "; ".join(mesh.get("minor") or []),
                    "impression": (rec.get("impression") or "").strip().replace("\n", " "),
                })
                taken.add(uid)
            cache[label] = fresh
            with open(QUERY_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1)
            source = ""
        else:
            source = " (banked)"

        # Dedupe across queries at assembly time, not at fetch time, so what is cached
        # for one query does not depend on which other queries had run first.
        got = 0
        for rec in cache[label]:
            if rec["uid"] in seen:
                continue
            rows.append(rec)
            seen.add(rec["uid"])
            got += 1
        print(f"  {label:15s} {got:3d} records{source}", flush=True)
        if source:
            continue
        # /api/search tolerates roughly one call per cooldown window: measured by
        # probing it, one request succeeded and the next five were all refused. Nine
        # calls at this spacing is about nine minutes, against an hour of GPU time
        # downstream, so patience here is nearly free.
        time.sleep(QUERY_SPACING)
    return rows, failed


def download_images(rows):
    """One download per manifest row. Returns (already_present, downloaded, failed).

    Files already on disk are skipped, so a throttled run resumes where it stopped
    instead of starting over.
    """
    # Checking the magic bytes rather than merely that a file exists. A previous run
    # left four HTML error pages on disk with .png names; a size check called them
    # present and they would never have been re-fetched.
    todo, corrupt = [], 0
    for row in rows:
        path = os.path.join(OUT, row["file"])
        if not os.path.isfile(path):
            todo.append(row)
            continue
        with open(path, "rb") as f:
            if looks_like_image(f.read(8)):
                continue
        corrupt += 1
        os.remove(path)
        todo.append(row)
    have = len(rows) - len(todo)
    print(f"\nimages: {have} already on disk, {len(todo)} to download", flush=True)
    if corrupt:
        print(f"  discarded {corrupt} file(s) that were not images at all", flush=True)

    got, failed, t0 = 0, [], time.time()
    for n, row in enumerate(todo, 1):
        try:
            # Shorter backoff than the search endpoint: this one tolerates sustained
            # use, and 500 downloads cannot each afford a 30s first retry.
            blob = get(BASE + row["img"], want="image", backoff=10.0)
        except Exception as exc:
            print(f"  ! {row['file']}: {exc}", flush=True)
            failed.append(row["file"])
            continue
        with open(os.path.join(OUT, row["file"]), "wb") as f:
            f.write(blob)
        got += 1
        time.sleep(IMAGE_SPACING)
        if got and got % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {got} downloaded, {len(todo) - n} to go "
                  f"(~{elapsed / got * (len(todo) - n):.0f}s left)", flush=True)
    return have, got, failed


def report_balance(rows):
    """Class balance per finding, before any GPU time is spent on the set.

    A finding with too few positives cannot support a threshold estimate, and it is
    far cheaper to learn that here than after an hour of inference.
    """
    print(f"\nclass balance ({len(rows)} images), ground truth from MeSH labels:")
    print(f"  {'finding':18s} {'n+':>4s} {'n-':>5s} {'prevalence':>11s}")
    thin = []
    for finding in FINDINGS:
        pos = sum(1 for r in rows if is_positive(r["problems"], finding))
        flag = ""
        if pos < MIN_POSITIVES:
            flag = f"   <-- under {MIN_POSITIVES}, will not be measured"
            thin.append(finding)
        print(f"  {finding:18s} {pos:4d} {len(rows) - pos:5d} "
              f"{pos / len(rows):10.1%}{flag}")
    if thin:
        print(f"\n  {len(thin)} finding(s) too thin to measure: {', '.join(thin)}")
        print("  analyze_reliability.py will leave them out of RELIABILITY rather than"
              "\n  emit a threshold that is really sampling noise.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-query Open-i even if MANIFEST.csv already exists")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write GROUND_TRUTH.csv even when queries or downloads failed")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if os.path.exists(MANIFEST) and not args.refresh:
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"manifest: {len(rows)} records already pinned "
              f"({os.path.relpath(MANIFEST)}); --refresh to re-query")
    else:
        # --refresh means re-ask Open-i, so the banked answers go too. Without it, a
        # re-run resumes at the first query that has not succeeded yet.
        if args.refresh and os.path.exists(QUERY_CACHE):
            os.remove(QUERY_CACHE)
        print(f"querying Open-i for {len(QUERIES)} findings...")
        rows, failed_queries = collect_metadata()
        # The metadata phase is nine cheap calls, so it is all-or-nothing: a partial
        # manifest that looks authoritative is the failure this script exists to avoid.
        if failed_queries and not args.allow_partial:
            print(f"\n!! {len(failed_queries)} of {len(QUERIES)} queries failed: "
                  f"{', '.join(failed_queries)}")
            print("Nothing written. Re-run to retry -- the metadata phase is cheap.")
            return 1
        if not rows:
            print("\n!! no metadata retrieved; nothing to do")
            return 1
        with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nmanifest: {len(rows)} records -> {os.path.relpath(MANIFEST)}")

    have, got, failed_images = download_images(rows)
    on_disk = [r for r in rows if os.path.isfile(os.path.join(OUT, r["file"]))]
    print(f"\n{len(on_disk)}/{len(rows)} manifest rows have an image "
          f"({have} kept, {got} new, {len(failed_images)} failed)")

    # GROUND_TRUTH.csv is what every downstream measurement trusts. Overwriting a good
    # one with a partial run is exactly how the previous evaluation set was lost.
    if failed_images and not args.allow_partial:
        print(f"\n!! {len(failed_images)} image(s) failed to download.")
        print(f"{os.path.relpath(GROUND_TRUTH)} NOT written, so an existing one is left"
              " intact.\nRe-run to retry just the gaps, or pass --allow-partial to "
              "accept this set as it stands.")
        return 1

    with open(GROUND_TRUTH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GROUND_TRUTH_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(on_disk)
    print(f"\n{len(on_disk)} images -> {os.path.relpath(GROUND_TRUTH)}")
    report_balance(on_disk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
