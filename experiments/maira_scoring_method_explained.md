# How we score MAIRA-2 for condition detection, and why the method holds up

Date: 2026-10-24. Covers how the CheXagent vs MAIRA-2 detection comparison scored MAIRA-2, a
concern we found with that score, and the checks we ran on it. All numbers come from the results
files listed at the end; all images are from NIH ChestX-ray14.

---

## 1. The problem: MAIRA-2 has no confidence score

To compare two models with AUC, each model has to give a **score** per image and condition: a
number that is higher when the model thinks the condition is more likely. AUC then measures how well
those scores rank images *with* the condition above images *without* it (0.5 = chance, 1.0 = perfect).

- **CheXagent** answers questions. We ask *"Is there pleural effusion?"* and use the probability it
  gives to the answer "Yes" (between 0 and 1). This is a native score.
- **MAIRA-2** cannot answer questions. It was built to do one thing: write the findings section of a
  radiology report. It has no yes/no mode and no confidence output.

So we had to get a score out of a model that only writes reports.

---

## 2. The original method: sentence likelihood

### The idea in one line
Instead of letting MAIRA-2 write, we ask it how likely it would be to **begin** its findings with
*"There is a pleural effusion."* versus *"There is no pleural effusion."*, and use the share going
to the first sentence as the score.

### Step by step

**Step 1: MAIRA-2 gets the image and its standard report prompt** (exactly as it receives it):

> You are an expert radiology assistant tasked with interpreting a chest X-ray study.
> USER: Given the current frontal image only `[image]` Provide a description of the findings in the
> radiology study in comparison to the prior frontal image. INDICATION: N/A TECHNIQUE: N/A
> COMPARISON: N/A
> ASSISTANT:

**Step 2: we do not let it write.** A language model writes one token (a word or word piece) at a
time. At each step it gives a probability for every possible next token. Normally it picks the
likeliest one and continues. Our code instead **supplies the tokens of a sentence we choose** and
records the probability MAIRA-2 gave each one. This is standard practice for scoring text with a
language model, often called *teacher forcing*. No training and no prompt change are involved.

The two test sentences, as MAIRA-2 sees them:

```
A:  There | is | a  | ple | ural | eff | usion | .
B:  There | is | no | ple | ural | eff | usion | .
```

The probability of a whole sentence is the product of its token probabilities:
P("There" | image) × P("is" | "There") × P("a" | "There is") × ... We add logarithms instead of
multiplying, because the numbers are very small.

**Step 3: turn the two probabilities into one score.**

> **score = P(A) / (P(A) + P(B))**

The two sentences differ only in "a" vs "no", so the score is essentially MAIRA-2's answer to:
*"Having seen this X-ray and begun 'There is...', would you rather continue with 'a' or 'no'?"*
That is as close to a yes/no question as a report-only model allows.

### A real example (image 00000096_007.png, NIH labels pleural effusion)

| | CheXagent | MAIRA-2 |
|---|---|---|
| What we asked | "Is there pleural effusion?" | likelihood of A vs B as its first sentence |
| Raw output | "Yes" with probability 0.948 | log-probability -20.6 for A, -16.5 for B |
| Score | **0.948** | 1 / (1 + e^(20.6 - 16.5)) = **0.015** |

The same pair was used for all 13 conditions (e.g. *"There is a mass." / "There is no mass."*).
That is 26 short sentences per image, all scored after a single pass over the image.

### Why a score like 0.015 can still detect
AUC looks only at **ranking**, not at the size of the number. MAIRA-2 nearly always prefers
"There is no...", so its scores are small. What matters is whether images with the condition score
higher than images without it. Here are pleural effusion scores on five development images:

| image | NIH label | MAIRA-2 score | CheXagent score |
|---|---|---|---|
| 00000096_007 | effusion | 0.015 | 0.948 |
| 00000231_004 | effusion | 0.047 | 0.965 |
| 00000090_008 | effusion | 0.001 | 0.555 |
| 00000006_000 | no effusion | 0.001 | 0.050 |
| 00000029_000 | no effusion | 0.000 | 0.029 |

---

## 3. The concern we found

We let MAIRA-2 write its normal report for image 00000096_007. It says:

> *"...Cardiac silhouette is enlarged, and is accompanied by mild pulmonary vascular congestion.
> **Small pleural effusions are present bilaterally.**"*

So MAIRA-2 **does** see the effusion, even though our score was only 0.015. Two features of the
method explain this:

1. **Wording.** MAIRA-2 writes "Small pleural effusions are present bilaterally", not "There is a
   pleural effusion." We tested a sentence it would not naturally write.
2. **Position.** We score the sentence as the *first* sentence of the findings. MAIRA-2 often opens
   with something else ("The patient is status post median sternotomy...") and mentions the
   effusion later.

A first check on 200 development images (100 with effusion, 100 without) supported the concern:

- MAIRA-2's own reports mentioned effusion in 65 of 100 labelled effusions, with 4 false alarms.
  That is close to CheXagent's 68 and 4 at its cut-off.
- The original score matched what MAIRA-2 wrote only partly (AUC 0.807 for its own statement).

This raised a fair question: **did our scoring method, rather than MAIRA-2 itself, cause MAIRA-2's
lower AUC?**

---

## 4. The test: three alternative scores

On **300 random NIH development images**, we compared the original score with three alternatives.
The test images used in the main comparison were not touched. MAIRA-2's phrasings were learned from
**another 300 development images**, so no method was tuned on the images it is judged on.

### Alternative A: MAIRA-2's own phrasings
We let MAIRA-2 write reports on the 300 phrasing images and collected the sentences it actually
writes about each condition. Up to 5 of the most frequent sentences stating the condition and 5
negating it are kept, plus the original pair. Real examples for pleural effusion:

| MAIRA-2 states effusion (times written) | MAIRA-2 negates effusion (times written) |
|---|---|
| "Small pleural effusions are present bilaterally." (3) | "No pleural effusion is evident on this single projection." (5) |
| "Small right pleural effusion is also demonstrated." (2) | "No pleural effusion or acute skeletal findings." (3) |
| "Small pleural effusions are also demonstrated." (2) | "There is no evidence of pleural effusion or pneumothorax." (3) |

Score = combined probability of all "states" sentences vs all "negates" sentences.

### Alternative A-strict: the cleaner phrasings only
A, keeping only phrasings that MAIRA-2 wrote at least twice and that mention no other condition and
no device. Some mined sentences were mostly about something else, e.g. *"Right internal jugular
vascular catheter terminates in the lower superior vena cava, with no visible pneumothorax"*. That
sentence partly measures whether a catheter is present. This rule was fixed after seeing the
phrasings but before any evaluation result.

### Alternative B: let MAIRA-2 write 10 times and count
MAIRA-2 writes 10 reports per image, sampling from its own probabilities (temperature 1) so that
each report can differ. The score is the share that mention the condition: 7 of 10 gives 0.7. This
measures exactly what MAIRA-2 would say, with no assumptions about wording or sentence order. It
is a standard way to get confidence from models that only produce text.

Real example, image 00005273_003.png (NIH: effusion). All 10 sampled reports mention effusion:
- *"Small-to-moderate left pleural effusion is present as well as adjacent left basilar atelectasis or consolidation."*
- *"Moderate sized left pleural effusion is present as well as a smaller right pleural effusion."*
- *"Increasing left pleural effusion with adjacent atelectasis."*
- ... and 7 more, so **B = 10/10 = 1.0**. The original score for this image was only **0.196**,
  though that still ranked it **8th of 300 images**.

A simple rule reader decided whether each report states a condition: the condition's term with no
negation word ("no", "not", "without", ...) before it. The same rule was applied to every method.

### How the methods were judged
1. **Faithfulness (no labels used):** does the score track what MAIRA-2 itself writes in its normal
   report? (AUC for MAIRA-2's own statement.)
2. **Detection:** does the score find what NIH labels? (AUC for the NIH label, the measure used
   in the CheXagent comparison.)

---

## 5. Results

Averages over the 5 conditions with at least 10 positive images in the 300 (pleural effusion,
atelectasis, infiltration, nodule, mass):

| MAIRA-2 score | Faithfulness (tracks what it writes) | Detection (AUC vs NIH labels) |
|---|---|---|
| **Original sentence pair** | 0.883 | **0.828** |
| A: own phrasings | 0.838 | 0.685 |
| A-strict: cleaner phrasings | 0.878 | 0.761 |
| B: share of 10 sampled reports | **0.959** | 0.777 |
| Its single normal report, yes/no | - | 0.654 |
| *CheXagent "Is there X?" (reference)* | | *0.863* |

Selected conditions (detection AUC):

| condition (positives) | original | A | B | CheXagent |
|---|---|---|---|---|
| pleural effusion (13) | 0.864 | 0.893 | **0.900** | 0.923 |
| atelectasis (13) | **0.871** | 0.546 | 0.795 | 0.937 |
| nodule (14) | **0.896** | 0.831 | 0.845 | 0.942 |
| mass (12) | **0.887** | 0.534 | 0.761 | 0.899 |
| infiltration (42) | **0.624** | 0.624 | 0.583 | 0.614 |

**What the results show:**
1. **B tracks MAIRA-2's writing best** (0.959), as designed.
2. **Tracking its writing does not detect disease better.** The original score has the highest
   average detection AUC. The alternatives help only for **pleural effusion** (0.864 to 0.900),
   the very condition that prompted the concern.

---

## 6. Why we believe the original score is still good

**MAIRA-2 often notices a finding without writing it down.** Its report is a short summary, and it
leaves things out. Asking it directly about each condition, *"There is a mass."* vs *"There is no
mass."*, reads its internal judgement, including findings it would not mention.

Real example, image 00013094_000.png (NIH: mass, atelectasis, infiltration):
- MAIRA-2's normal report mentions sternotomy, aortic elongation, an elevated left hemidiaphragm
  and increased density at the left base, but **no mass**.
- **None of its 10 sampled reports mention a mass** (B = 0/10).
- Yet the original score for "There is a mass." was **0.784**, the **6th highest of 300 images**
  (median 0.002).

Scores that track MAIRA-2's writing lose this hidden information; the direct question keeps it.
This is why the original score detects best, even though it matches the wording least.

**In summary:**
1. **The original score is the best of the four ways we tried** to measure MAIRA-2's detection.
   Its average detection AUC is 0.828, against 0.685-0.777 for the alternatives.
2. **The CheXagent vs MAIRA-2 comparison was not unfair to MAIRA-2 overall.** None of the
   better-fitting scores closed the gap to CheXagent on average.
3. **One honest exception:** for pleural effusion the original score modestly understates MAIRA-2
   (0.864 vs 0.900 with sampled reports).
4. **The method was pre-registered.** It was fixed before the main comparison's readings
   (`maira_score_design.md`). This check supports it after the fact; it does not replace it.

---

## 7. Limitations

- **Small numbers.** Only 5 conditions have at least 10 positives in 300 images (e.g. effusion 13,
  mass 12). Differences between methods are uncertain, and no confidence intervals were computed
  for this check.
- **Exploratory.** This check was run after the main results were known, on development images
  only. The A-strict rule was fixed after seeing the phrasings.
- **The rule reader is simple.** It decides "states the condition" by keywords and negation words,
  and can misread complex sentences. For example, *"pneumonia with or without a component of
  pulmonary edema"* was counted as negating edema.
- **Detection, not description.** All scores measure whether a condition is present, not where it
  is or how it has changed. Grounding (box location) was assessed separately: CheXagent's box side
  was informative, MAIRA-2's was not.
- **NIH labels** are text-mined from reports and contain errors.
- **Possible NIH advantage for CheXagent.** CheXagent was probably trained on public NIH images;
  MAIRA-2 was not.

---

## Files

| What | File |
|---|---|
| Original score (code) | `experiments/maira_score_nih.py` (lines 38-75) |
| Original score design (pre-registered) | `experiments/maira_score_design.md` |
| Main comparison, all 13 conditions | `experiments/results/maira_vs_chexagent_all_conditions.md` |
| First check (effusion, 200 images) | `experiments/results/maira_score_validity.md`, `experiments/maira_score_validity.py` |
| This method comparison (300 images) | `experiments/results/maira_score_methods_dev.md`, `experiments/maira_score_methods_dev.py` |
| MAIRA-2's mined phrasings | `experiments/results/maira_methods_phrasings.json` |
