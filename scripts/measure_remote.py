"""Add a served model's answers to an existing reliability.json.

measure_reliability.py runs the three in-process tools over the evaluation set. This
adds a model that lives behind a service, in its own environment, without re-running
the ones already measured -- the 544 films and their CheXagent, classifier and report
answers are kept exactly as they are, and one column is appended.

That separation is what makes the split-environment design pay off twice: MedGemma
cannot be imported into this venv at all (it needs transformers>=4.50 and torch>=2.6,
this one has 4.40 and 2.5.1), but it can be measured from it over HTTP.

    python -m medrax.serve.server --backend medgemma --port 8102   # in its own venv
    python scripts/measure_remote.py --url http://127.0.0.1:8102   # from this one

The column is named after the service's own tool name, so analyze_reliability.py
discovers it without being told. Resumable: answers already recorded are skipped, so
an interrupted sweep costs at most one image.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from findings import FINDINGS, is_positive  # noqa: E402
from medrax.tools.remote import RemoteModelTool  # noqa: E402

DATA = os.getenv("MEDRAX_EVAL_DIR", "data/indiana_eval")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="the model service, e.g. http://127.0.0.1:8102")
    parser.add_argument("--input", default="reliability.json")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="P(yes) is read off the first token; the explanation after "
                             "it costs time and is never used (default 8)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N images, for a smoke test")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    tool = RemoteModelTool.from_health(args.url)
    field = f"{tool.name}_p"
    print(f"measuring {tool.name} ({tool.model_id}) -> field {field!r}", flush=True)

    with open(args.input, encoding="utf-8") as handle:
        records = json.load(handle)

    todo = [r for r in records
            if any(field not in r["findings"].get(f, {}) for f in FINDINGS)]
    if args.limit:
        todo = todo[:args.limit]
    calls = len(todo) * len(FINDINGS)
    print(f"{len(records)} images, {len(todo)} still to do, ~{calls} calls", flush=True)
    if not todo:
        print("nothing to measure; every image already has this column")
        return 0

    start, done = time.time(), 0
    for index, record in enumerate(todo, 1):
        image = os.path.join(DATA, record["file"])
        if not os.path.isfile(image):
            continue
        for finding in FINDINGS:
            entry = record["findings"].setdefault(finding, {})
            if field in entry:
                continue
            # The same template the reliability table was built on, so the new column
            # is comparable with the existing ones rather than measuring a different
            # question that happens to share a name.
            payload, _ = tool._call(image, f"Does this chest X-ray contain a {finding}?",
                                    max_new_tokens=args.max_new_tokens)
            entry[field] = payload.get("confidence")
            entry.setdefault("truth", is_positive(record.get("problems", ""), finding))
            done += 1
        if index % 10 == 0:
            elapsed = time.time() - start
            rate = elapsed / max(done, 1)
            left = (len(todo) - index) * len(FINDINGS) * rate
            with open(args.input, "w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=1)
            print(f"  {index}/{len(todo)} images  {elapsed:.0f}s  ~{left:.0f}s left", flush=True)

    with open(args.input, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)
    answered = sum(1 for r in records for f in FINDINGS
                   if r["findings"].get(f, {}).get(field) is not None)
    print(f"\nDONE {done} calls in {time.time() - start:.0f}s -> {args.input}")
    print(f"{answered} of {len(records) * len(FINDINGS)} values carry a usable probability")
    if answered < done:
        print("values came back None where the model did not answer with yes or no; "
              "those are dropped rather than scored, by design")
    print("\nNext: python scripts/analyze_reliability.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
