# Plan to strengthen the manuscript (2026-10-25)

## Positioning (agreed)
Not "the first method to verify presence and location": Mahmood et al. (MICCAI 2025, "Phrase-grounded
fact-checking for automatically generated chest X-ray reports") already verify findings and their
locations against the image, with a dedicated trained model, on ChestX-ray8, VinDr-CXR, MS-CXR and
Chest ImaGenome. Our contribution is a controlled investigation of:
1. **claim conditioning**: how much the evidence returned by existing checkers depends on what the
   prompt asserts (laterality now; presence to be added);
2. **evidence independence**: collecting evidence without the assertion, and how informative it is;
3. **usefulness of existing, off-the-shelf checkers** (CheXagent, MAIRA-2) as evidence sources, with
   calibrated, frozen decision rules and pre-registered evaluation.
We do not claim that Mahmood's dedicated model shares the failure mode: we have not tested it.

### What the Mahmood paper does (from the MICCAI open-access PDF)
- Fact-checking model: chest X-ray CLIP image encoder + text encoder, retrained with a supervised
  multi-label contrastive loss on real vs fake finding labels, plus a regression head that outputs a box
  (x, y, w, h) and a veracity label E.
- Training data: about 27M synthetic image-finding pairs from ground-truth reports, perturbed by
  polarity reversal, relocation to another valid location, and substitution.
- Inference: findings extracted from a generated report, in a structured form, are given with the image.
  The model predicts veracity and location, and the location is compared with the report's indicated
  location in an error score (RQ).
- Results: real/fake accuracy 0.90-0.94; mIoU 0.49-0.57 (MAIRA-2 0.39-0.51, Med-RPG 0.23-0.38);
  concordance 0.997 with ground-truth-based error scores across seven report generators.
- **Unclear** from the paper: whether the stated location is part of the model's text input. Fig. 1c
  compares the predicted location with the indicated location after prediction, but the relocation
  perturbations imply that location is in the input. The stated polarity ("yes"/"no") is in the input,
  so presence checking is claim-visible.
- No left/right-specific analysis, no calibration to likelihood ratios, no pre-registration. Code and
  data are described as "being contributed to open source"; no URL was found.

### Manuscript changes for positioning
- A Related Work paragraph on Mahmood et al. with the facts above, and a column for it in Table 1.
- Contributions and abstract worded as "controlled investigation of claim conditioning and evidence
  independence in existing checkers", never "first to verify presence and location".
- Discussion: our results concern general-purpose checkers; dedicated fact-checkers trained on
  perturbed claims are a different design whose conditioning behaviour is untested.
- Add the other required citations from `related_work_survey.md`: RadAgents (side-free MAIRA-2 with
  post-hoc side), CoEV, UniVRSE, CorBenchX, Verification Mirage, Source-Dependent Deference,
  Counterfactual REC, HEAL-MedVQA, CONRep, Chain-of-Verification.

## Work plan, in priority order
Each experiment gets a protocol committed before its outputs are generated, as before.

### Tier A: the core claim (claim conditioning and evidence independence)
**A1. Presence conditioning (new, highest value).** Does telling the checker what the report asserts
change its presence evidence?
- Arms per image and condition: neutral ("Is there {c}?"); assertion-positive ("The report states
  there is {c}. Is there {c}?"); assertion-negative ("The report states there is no {c}. Is there {c}?").
- Data: NIH test split (1,500 images, 13 conditions; existing neutral readings) and VinDr (3,000).
- Checkers: CheXagent; MedGemma 1.5 4B as a checker; one recent grounded model (RadVLM or CheXOne) if
  it runs locally.
- Measures: shift in p_present; AUC under each arm; rate at which a false assertion flips the decision;
  LR+ and LR- of the conditioned score.
- This extends the result from laterality to presence, which is what report-blinding mainly protects.

**A2. Laterality conditioning, broader.**
- More models: MedGemma 1.5 grounding, RadVLM or CheXOne.
- More data: all one-sided findings in a larger VinDr sample (by amendment); MS-CXR phrases with a
  side; ChestX-Det if boxes allow.
- Control: horizontally flipped images with swapped truth, to separate image evidence from an
  anatomical or positional prior (RadVLM fails on flips in AnatomiX).

**A3. Test an EDV-style validator directly.**
- Use CXRAgent's validator prompt (`_explain_func`, commit 0514b489) with an open multimodal model
  (Qwen2.5-VL or MedGemma), since the paper uses GPT-4o / Qwen-VL-Max.
- Give it a tool statement that is true or false (presence and side), versus a blinded version that
  asks for evidence about the condition without the statement.
- Measures: agreement with the statement when false vs true; credibility rating by truth.
- This replaces "we did not test EDV" with a direct result.

**A4. Image-ablation controls.**
- Blank, noise and swapped-patient images for p_present and grounding. This shows the evidence comes
  from the image (as raised by Lotfinia 2026).
- We already fixed one CheXagent image-ignored bug, so this is cheap reassurance.

**A5. Verification task with known truth (makes "usefulness" concrete).**
- Build statements with known truth from labels and boxes: true and false presence statements, and
  true and false side statements (side swaps).
- Compare report-blinded evidence, statement-given evidence (from A1 and A2), and Mahmood's model if it
  becomes available, by AUC for separating true from false statements.
- This is the cleanest test of whether evidence independence improves verification, and it does not
  depend on the report reader.

**A6. Mahmood's model.** Ask the authors for code and weights. If they are obtained, run it on the A5
statements (and its conditioning, if location is an input). If not, state that it is untested.

### Tier B: downstream credibility
- **B1. Reader validation:** annotate the 90-report sheet (protocol ready; needs one or two
  annotators, ideally with radiology training).
- **B2. Report-level error flagging:** inject controlled errors (false finding, omitted finding, side
  swap) into MedGemma reports, as in ReXErr or CorBenchX, and measure flagging by blinded checking.
- **B3. Reporting:** report fusion as a negative result for CheXagent. Optionally test a
  development-fitted combination (logistic on development data) as exploratory.

### Tier C: the fine-tuned MedGemma (secondary contribution, or a separate paper)
- **C1.** Base MedGemma 1.5 in bf16 with the identical input format on the same 499 NIH pairs. This
  removes the quantisation and format confounds; no new data needed.
- **C2.** SFT-only checkpoint vs SFT+GRPO on NIH. ChexReason found that GRPO hurt NIH transfer. The
  SFT-only checkpoint is needed from the team.
- **C3.** No-prior and shuffled-prior inputs. These test whether change statements depend on the
  actual prior, since hallucinated comparisons are a known failure (Banerjee 2024).
- **C4.** Change correctness on held-out MIMIC-CXR: MS-CXR-T progression labels, the L-MIMIC
  annotator or STRIVE's LCC, plus RadGraph-F1, CheXbert F1 and GREEN. Baselines: MAIRA-2 with prior,
  Libra, CheXOne, base MedGemma 1.5. Needs PhysioNet-credentialed access, the images, and
  confirmation that the held-out set was not used for checkpoint selection.
- **C5.** Training documentation from the team: data size and patients, split, reward function,
  hyperparameters, compute, checkpoint selection.

### Tier D: writing and reporting
- **D1.** Title and framing. Lead with claim conditioning and evidence independence. Present the
  fine-tuned MedGemma as secondary until C1-C4 exist, or split it into its own paper.
- **D2.** Related Work rewrite (Mahmood paragraph, survey citations); verify every reference.
- **D3.** STARD-AI checklist (primary), CLAIM 2024, and TRIPOD-LLM items for prompts and decoding,
  in a supplement.
- **D4.** Public repository with commit history and protocol files (and an OSF snapshot) so the
  pre-registration dates can be checked.
- **D5.** Pre-empt contradicting grounding rankings: CheXagent ranks last on anatomy-query benchmarks.
  Explain the difference in tasks and prompts.

## Suggested sequencing
1. D2 (Mahmood paragraph), A4 and C1: quick, no new data or collaborators.
2. A1 and A5 protocols, then runs: the main new evidence.
3. A2 and A3: broaden the core result.
4. B1 as soon as an annotator is available; B2 after A5.
5. C2-C5 when the team provides the checkpoints, details and MIMIC access.
6. D1, D3, D4 and a full rewrite once Tier A results are in.

## Target: a medical journal (decided 2026-10-25)

### Shortlist (check each journal's current author guidelines before formatting)
| Journal | Fit | What it will expect |
|---|---|---|
| Radiology: Artificial Intelligence (RSNA) | Best fit: evaluation of imaging AI, laterality and safety framing | CLAIM checklist; radiologist co-authors; rigorous statistics; clinically meaningful endpoints |
| Journal of Imaging Informatics in Medicine | Good fit; similar work (CONRep) published there | Clear methods; reproducibility |
| European Radiology / Insights into Imaging | Clinical radiology readership | Clinical relevance; reader involvement |
| JAMIA / JMIR Medical Informatics | Informatics and agent-validation angle | Evaluation design; generalisability |
| npj Digital Medicine / Lancet Digital Health | Higher bar | Clinical impact, ideally a reader study |

### What changes for a medical journal
1. **Clinical framing first.**
   - The safety question: when an AI draft states a finding and a side, can an independent image check
     be trusted, or does it echo the draft?
   - Wrong-side statements are a recognised patient-safety error. Cite the laterality-error literature
     (JACR 2024 laterality study; JMIR Formative Research 2025; CorBenchX side-confusion errors).
2. **Radiologist involvement.**
   - At least one radiologist co-author.
   - The reader-validation annotation (B1) done by a radiologist.
   - Ideally a small reader study:
     - design: radiologists review AI drafts with and without report-blinded evidence flags;
     - measures: errors caught, false alarms, reading time.
   - This is the single biggest upgrade for a medical journal.
3. **Endpoints clinicians read easily.**
   - Sensitivity and specificity for catching false findings and wrong-side statements (A5), with
     likelihood ratios (already familiar from diagnostic medicine).
   - Report these in preference to mean-AUC differences.
4. **Reporting.**
   - CLAIM 2024 (required by Radiology: AI) and STARD-AI checklists.
   - Ethics statement (public de-identified data; IRB exemption if applicable).
   - Data and code availability.
   - Register the remaining protocols on OSF as well as in git.
5. **Shorter main text.**
   - Main paper: laterality conditioning, presence conditioning (A1), the verification test (A5), and
     the checker comparison.
   - Supplement: engine details, MAIRA-2 scoring, fusion, grounding detail, and the full
     pre-registration log.
   - Structured abstract, key points and summary statement as the journal requires.
6. **Fine-tuned MedGemma.**
   - A medical journal would judge it on clinical report quality, which needs radiologist assessment of
     interval-change statements (C4) and the clean baseline (C1).
   - Keep it as a secondary result, or submit it separately once evaluated.

### Revised priority for the journal route
1. A1 presence conditioning and A5 verification test (the main new evidence).
2. Radiologist recruitment: co-author, reader validation (B1), then a small reader study.
3. A4 image-ablation controls and C1 clean baseline (quick).
4. A2 broader laterality and A3 EDV-style validator.
5. Rewrite to the journal format with CLAIM and STARD-AI.

## Venue fit (if an ML venue is considered instead)
- With Tier A done: a full paper at MICCAI or MIDL, or a clinical-AI journal (Radiology: AI, with
  CLAIM).
- Current state: a workshop paper (ML4H, a MICCAI workshop).
- The fine-tuned MedGemma alone, with C1-C4, could be a separate short paper.
