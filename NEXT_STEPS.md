# Next steps: Evidence Engine v1.0 (status as of 2026-09-27)

Branch `medgemma-edv`. Nothing on this branch has been pushed.

## Where the work stands

| Step | Result | Record |
|---|---|---|
| Indiana discovery, 544 images | p_present ranks findings well (AUC 0.80–0.96) but is not a probability | tag `ppresent-validation-v1` |
| Indiana hold-out (70/30) | Platt calibration helps 5/6 findings; per-finding thresholds shift the operating point | `experiments/results/ppresent_split.md` |
| NIH ChestX-ray14 external validation (A/B/C) | discrimination transfers; calibration is population-dependent | `experiments/results/ppresent_nih.md` |
| D1 / D2 | prevalence shift explains 87–99% of the Indiana → NIH calibration gap; SUPPORTING transfers, REFUTING is fragile in the score tails | commits `47c6b64` (protocol), `9871ee8` (results) |
| D3 | dual-evidence REFUTING rule removes most tail failures | commits `aee83a3` (protocol), `3643e42` (results) |
| **Evidence Engine v1.0 frozen** | NIH-derived Platt LR + NIH-development empirical bands; Indiana comparator; runtime reproduces D2/D3 on 60,480 assessments | tag **`evidence-engine-v1.0`** (`c30c5d7`), spec `medrax/evidence/EVIDENCE_ENGINE_V1.md` |
| Final external validation | **pre-registered, not yet run** (data not downloaded) | commit `5280b43` |

## The one remaining step: VinDr-CXR external validation

This is the **single, final experiment** for Evidence Engine v1.0. Its protocol and code were
committed before the data was on this machine:
`experiments/vindr_external_validation_design.md` and
`experiments/vindr_external_validation.py`.

1. **Get the data** (needs a credentialed PhysioNet account and the signed data use
   agreement): https://physionet.org/content/vindr-cxr/1.0.0/
   Download only the test split and its labels, not the full ~200 GB release:
   - `test/` (3,000 DICOMs)
   - `annotations/image_labels_test.csv`

   Place them under `D:\datasets\vindr-cxr\` keeping the `test/` and `annotations/`
   folders, or pass another location with `--data` (or set `VINDR_DIR`).
2. **Run** from the repository root, in the project venv:
   ```powershell
   python experiments/vindr_external_validation.py prepare   # checks files, converts DICOMs, freezes the manifest
   python experiments/vindr_external_validation.py run       # CheXagent readings, ~90 min on an RTX 3090, resumable
   python experiments/vindr_external_validation.py analyze   # pre-registered endpoints -> experiments/results/vindr_external_validation.md
   ```
   `prepare` stops instead of guessing if the files do not match the protocol (for example,
   several label rows per image) and reports label counts only. `analyze` refuses to run on
   incomplete readings and verifies the engine's checksums on load.
3. **Commit** the manifest, readings and report, and write up the result.

### Rules for this step (do not relax)
- The engine is used exactly as tagged: no threshold, calibration or band fitting, no state
  redesign, no finding-specific exceptions.
- Run once. Report the result whatever it is; partial success is the result, and failures
  go to Limitations/Discussion.
- Report both engines (NIH primary, Indiana comparator); do not select one on VinDr results.
- **No D4, no v1.1, no further dataset for this version.** After this step: stop and write
  the paper.

### Before running, check
- The venv still has the versions in `experiments/results/REPRODUCE.md` (transformers
  commit `88d9609…`, torch 2.6.0+cu124) and the CheXagent snapshot `8f19b53a…`.
- `medrax/evidence/evidence_engine_v1.0.sha256` still matches (the engine checks this on
  load).
- `.gitattributes` keeps `experiments/results/**` and `medrax/evidence/**` byte-exact; keep
  it, or the committed checksums stop matching.

## Deliberately deferred (not part of this study)
- **Production pipeline integration** of the engine (compute p_present for every finding
  MedGemma states, show state / reason / LR in the report, handling of findings outside the
  six validated ones such as pneumonia, UI and localization). A later system phase.
- **Paper 2 question**: does the Evidence Engine improve verification of MedGemma's
  conclusions compared with EDV-style LLM critique? Needs MedGemma claims scored against
  ground truth; a separate experiment family.
- MedSAM, MAIRA-2, other localization tools, multi-agent designs, Qwen vs GPT-4o director
  comparisons.

## Paper framing
Paper 1: whether CheXagent's intrinsic signal can be converted into a transferable,
prevalence-separated evidence measure for chest X-ray findings. Suggested wording: "p_present
preserves substantial discriminative information across datasets, while the
likelihood-ratio representation separates much of the transferable evidence signal from
population-specific prevalence." Describe NIH as part of the engine's development (D3 used
NIH development data), not as external validation of the finished engine.
