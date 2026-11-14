# Grounding recall: where do CheXagent and MAIRA-2 put the box?

Date: 2026-10-24. Pre-registered experiment: protocol `experiments/grounding_recall_protocol.md`,
committed (`6106a1d`) before any box was produced; results committed unchanged (`cd3ba76`). All
images are from NIH ChestX-ray14.

---

## 1. Why we ran this

Our earlier comparison showed CheXagent ahead of MAIRA-2 at **detection**: telling *whether* a
condition is present (AUC of the "Is there X?" score). Our professor asked a different question:
**grounding recall**, i.e. when a model draws a box to show *where* a finding is, how often does that
box match where a radiologist marked it?

The two abilities are separate. A model can be good at saying "there is an effusion" and still draw
its box in the wrong place, or the other way round. This experiment measures grounding on its own.

---

## 2. Key terms

**Grounding.** The model is given a phrase (e.g. "pleural effusion") and an X-ray, and draws one or
more rectangles around where it thinks that finding is.

**IoU (Intersection over Union).** How much two boxes overlap:

> IoU = (area where the two boxes overlap) / (area covered by either box)

- 1.0 = the boxes are identical; 0 = they do not touch.
- Example: two equal squares where one is shifted sideways by half its width overlap on half of each
  square. IoU = 0.5 / 1.5 = **0.33**.
- IoU penalises a box that is too small, too large or misplaced.

**Recall at IoU t.** Of all the radiologist-marked findings, the share where the model's box
overlaps the radiologist's box with IoU of at least t. We report **t = 0.5** (the most common
standard; our primary measure) and **t = 0.3** (more lenient, often used for chest X-rays because
the radiologists' boxes are themselves approximate).

**Pointing game.** A hit if the *centre* of a model's box falls inside the radiologist's box. It
asks "did the model point at the right place?", ignoring box size. It is the most lenient measure.

---

## 3. Data

NIH ChestX-ray14 includes a set of images where radiologists drew a box around a finding
(`BBox_List_2017.csv`).
- We used every such box whose image we have, excluding images in our NIH test split (kept separate
  for other experiments, as in the earlier check).
- That gives **947 cases** (one image + one labelled finding + one radiologist box) on **847
  images**, covering 8 conditions.
- All NIH images are 1024 x 1024 pixels, so box coordinates can be compared directly.

| condition | cases |
|---|---|
| Atelectasis | 173 |
| Pleural effusion | 150 |
| Cardiomegaly | 141 |
| Infiltration | 117 |
| Pneumonia | 116 |
| Pneumothorax | 93 |
| Mass | 81 |
| Nodule | 76 |

---

## 4. How each model was asked

Both models were given **the condition name only, with no side** (e.g. "pleural effusion", not
"left pleural effusion"). An earlier test showed that both models simply box whichever side a
phrase names, even the wrong one. Each model was told the condition is there, so this measures
**location only, not detection**.

**CheXagent**
- Prompt: `Please locate the following phrase: pleural effusion`
- It answers with coordinates, e.g.
  `<|ref|> Pleural effusion <|/ref|> <|box|> (10,20),(47,85) <|/box|>`, where the numbers are
  percent of image width and height.
- We used the same function as our EDV pipeline (`locate_findings`). It ignores a box the model
  labels as a different finding.

**MAIRA-2**
- Its built-in phrase-grounding mode, given the phrase "Pleural effusion".
- It answers with text plus coordinates on a 100 x 100 grid, e.g.
  `<obj> Mass.<box><x26><y45><x38><y60></box></obj>`.
- We converted those to the same percent coordinates with MAIRA-2's own conversion function.
- Settings: full precision (bf16), pinned model version, deterministic output.

A model may return several boxes. For the main measures we use **its best box** (the one with the
highest IoU). A "union" check that penalises extra boxes is also reported. **No box counts as a
miss.**

---

## 5. Results

### 5.1 Pre-registered hypotheses (both two-sided)

| | CheXagent | MAIRA-2 | CheXagent minus MAIRA-2 [95% CI] | Result |
|---|---|---|---|---|
| **G1 (primary): recall at IoU >= 0.5** | 23.3% (221/947) | 19.0% (180/947) | **+4.3%** [+1.7% to +7.0%] | CheXagent better |
| **G2: recall at IoU >= 0.3** | 52.5% (497/947) | 30.3% (287/947) | **+22.2%** [+18.8% to +25.6%] | CheXagent better |

The confidence intervals come from 2,000 bootstrap resamples of images. McNemar tests on the cases
where the models disagree give the same conclusion:
- At IoU 0.5: CheXagent hit and MAIRA-2 missed in 103 cases, the reverse in 62 (p = 0.002).
- At IoU 0.3: 267 vs 57 (p < 10^-30).

### 5.2 All measures

| measure | CheXagent | MAIRA-2 |
|---|---|---|
| Recall at IoU >= 0.5 | 23.3% [20.8-26.1] | 19.0% [16.6-21.6] |
| Recall at IoU >= 0.3 | 52.5% [49.3-55.6] | 30.3% [27.5-33.3] |
| Pointing game | 68.5% [65.5-71.4] | 40.0% [36.9-43.2] |
| Union recall at IoU >= 0.5 (penalises extra boxes) | 23.2% | 18.7% |
| Union recall at IoU >= 0.3 | 49.3% | 29.0% |
| Mean IoU (best box) | 0.31 | 0.22 |
| Boxes per case (mean) | 1.26 | 0.92 |
| Box area / radiologist box area (median) | 0.91 | 1.25 |
| Cases with no box returned | 9 (1%) | 127 (13%) |

### 5.3 By condition

| condition | cases | recall@0.5 CheXagent / MAIRA-2 | recall@0.3 CheXagent / MAIRA-2 | pointing game CheXagent / MAIRA-2 | mean IoU CheXagent / MAIRA-2 |
|---|---|---|---|---|---|
| Atelectasis | 173 | **12%** / 2% | **42%** / 16% | 61% / 31% | 0.26 / 0.13 |
| Pleural effusion | 150 | 6% / 3% | 38% / 28% | 67% / 41% | 0.24 / 0.17 |
| Cardiomegaly | 141 | 70% / **100%** | 99% / 100% | 99% / 100% | 0.56 / **0.75** |
| Infiltration | 117 | **21%** / 11% | **58%** / 29% | 74% / 44% | 0.34 / 0.19 |
| Pneumonia | 116 | **17%** / 4% | **53%** / 12% | 78% / 16% | 0.31 / 0.08 |
| Pneumothorax | 93 | **33%** / 13% | **55%** / 22% | 68% / 32% | 0.34 / 0.17 |
| Mass | 81 | **21%** / 1% | **47%** / 7% | 63% / 26% | 0.29 / 0.08 |
| Nodule | 76 | 1% / 0% | 11% / 3% | 17% / 3% | 0.08 / 0.03 |

Bold marks a difference with McNemar p < 0.05 (per-condition tests are descriptive and not
adjusted for multiple comparisons). Effusion and nodule differences were not significant.

### 5.4 Examples

![Grounding examples](results/figures/grounding_examples.png)

Green = radiologist's box; blue = CheXagent; dashed orange = MAIRA-2.
- **A (mass):** CheXagent's box covers most of the radiologist's box (IoU 0.74). MAIRA-2's box is
  inside the mass area but far too small (IoU 0.08), so it counts as a miss.
- **B (cardiomegaly):** MAIRA-2's box matches the radiologist's heart box closely (0.87); CheXagent's
  is too short (0.47).
- **C (pneumothorax):** the pneumothorax is at the top of the patient's left lung. CheXagent finds
  it (0.60); MAIRA-2 boxes the patient's right (0.00). This matches an earlier finding: without a
  side named, MAIRA-2 tends to box the patient's right.
- **D (pleural effusion):** CheXagent boxes both sides with large boxes (best IoU 0.14). MAIRA-2's
  single box is closer (0.39). Both miss at IoU 0.5. NIH gives one box, even when an effusion may be
  on both sides.
- **E (mass):** MAIRA-2 returns no box, only the text "The cardiac silhouette is normal in size.
  There is no pneumothorax." CheXagent hits (0.61).
- **F (nodule):** both miss; nodules are small and neither model finds them reliably.

### 5.5 More examples, in both directions

To show where each model does better, we drew example figures **in both directions** with the same
fixed rule. One model's box hits the radiologist's box (IoU >= 0.5) and the other's misses (IoU <
0.3, or no box). Cases are ordered by the IoU gap, largest first, up to six per figure. These are
**selected examples**; the table gives how often each pattern occurred.

| condition | cases | CheXagent hits, MAIRA-2 misses (< 0.3) | MAIRA-2 hits, CheXagent misses (< 0.3) | at the IoU 0.5 line: only CheXagent hits / only MAIRA-2 hits |
|---|---|---|---|---|
| Atelectasis | 173 | 16 | 3 | 20 / 3 |
| Pleural effusion | 150 | 6 | 3 | 9 / 5 |
| Cardiomegaly | 141 | 0 | 1 | **0 / 42** |
| Infiltration | 117 | 11 | 2 | 18 / 7 |
| Pneumonia | 116 | 17 | 1 | 17 / 2 |
| Pneumothorax | 93 | 17 | 1 | 21 / 2 |
| Mass | 81 | 16 | 0 | 17 / 1 |
| Nodule | 76 | 1 | 0 | 1 / 0 |
| **All** | **947** | **84** | **11** | **103 / 62** |

The last column splits both models at IoU 0.5. Its totals (103 / 62) are the discordant cases of
the pre-registered McNemar test in 5.1.

**Why cardiomegaly needs its own rule.** CheXagent's heart boxes nearly always land in the right
place (recall at IoU 0.3: 99%). They usually overlap the radiologist's box by 0.3-0.5, so the "< 0.3"
rule finds only 1 case where MAIRA-2 does better. MAIRA-2's real advantage is at the 0.5 standard:
in 42 of 141 cases MAIRA-2 reaches IoU 0.5 and CheXagent does not, never the reverse. The
cardiomegaly figure therefore splits both models at 0.5.

#### Where MAIRA-2's box is better

| figure | cases shown | what it shows |
|---|---|---|
| `grounding_maira2_better_cardiomegaly_iou05.png` | 6 of 42 | MAIRA-2's box matches the radiologist's heart box (IoU 0.62-0.87); CheXagent's covers only the lower part of the heart (IoU 0.32-0.47) or is missing (1 case). |
| `grounding_maira2_better_effusion.png` | 3 of 3 | CheXagent's boxes are much smaller than the effusion; MAIRA-2's match its extent. |
| `grounding_maira2_better_atelectasis.png` | 3 of 3 | |
| `grounding_maira2_better_infiltrate.png` | 2 of 2 | |
| `grounding_maira2_better_pneumonia.png` | 1 of 1 | |
| `grounding_maira2_better_pneumothorax.png` | 1 of 1 | |
| `grounding_maira2_better_cardiomegaly.png` | 1 of 1 | (the "< 0.3" rule; CheXagent returned no box) |
| *(mass, nodule)* | 0 | no case where MAIRA-2 hits and CheXagent misses |

![MAIRA-2 better: cardiomegaly](results/figures/grounding_maira2_better_cardiomegaly_iou05.png)
![MAIRA-2 better: pleural effusion](results/figures/grounding_maira2_better_effusion.png)
![MAIRA-2 better: atelectasis](results/figures/grounding_maira2_better_atelectasis.png)
![MAIRA-2 better: infiltration](results/figures/grounding_maira2_better_infiltrate.png)
![MAIRA-2 better: pneumonia](results/figures/grounding_maira2_better_pneumonia.png)
![MAIRA-2 better: pneumothorax](results/figures/grounding_maira2_better_pneumothorax.png)

#### Where CheXagent's box is better

| figure | cases shown | what it shows |
|---|---|---|
| `grounding_chexagent_better_pneumothorax.png` | 6 of 17 | In all six the pneumothorax is at the top of the patient's left chest; CheXagent boxes it there, MAIRA-2 boxes the same region on the patient's right (the side bias seen in the VinDr side test). |
| `grounding_chexagent_better_mass.png` | 6 of 16 | CheXagent's box sits on the mass; MAIRA-2's is elsewhere, much too small, or missing (2 of 6). |
| `grounding_chexagent_better_pneumonia.png` | 6 of 17 | In 5 of the 6 (13 of all 17) MAIRA-2 returned no box; it did so for 74 of 116 pneumonia cases. |
| `grounding_chexagent_better_atelectasis.png` | 6 of 16 | |
| `grounding_chexagent_better_infiltrate.png` | 6 of 11 | |
| `grounding_chexagent_better_effusion.png` | 6 of 6 | |
| `grounding_chexagent_better_nodule.png` | 1 of 1 | Both models rarely box nodules correctly. |

![CheXagent better: pneumothorax](results/figures/grounding_chexagent_better_pneumothorax.png)
![CheXagent better: mass](results/figures/grounding_chexagent_better_mass.png)
![CheXagent better: pneumonia](results/figures/grounding_chexagent_better_pneumonia.png)
![CheXagent better: atelectasis](results/figures/grounding_chexagent_better_atelectasis.png)
![CheXagent better: infiltration](results/figures/grounding_chexagent_better_infiltrate.png)
![CheXagent better: pleural effusion](results/figures/grounding_chexagent_better_effusion.png)
![CheXagent better: nodule](results/figures/grounding_chexagent_better_nodule.png)

**Taken together:**
- MAIRA-2 draws the better box for **the heart**, and its boxes are sometimes better sized for
  effusion.
- CheXagent draws the better box for **most lung findings**, mainly because MAIRA-2 puts the box on
  the wrong side, makes it too small, or gives no box.

---|---|---|---|---|
| Atelectasis | 173 | 16 | 3 | `grounding_chexagent_better_atelectasis.png` |
| Pleural effusion | 150 | 6 | 3 | `grounding_chexagent_better_effusion.png` |
| Infiltration | 117 | 11 | 2 | `grounding_chexagent_better_infiltrate.png` |
| Pneumonia | 116 | 17 | 1 | `grounding_chexagent_better_pneumonia.png` |
| Pneumothorax | 93 | 17 | 1 | `grounding_chexagent_better_pneumothorax.png` |
| Mass | 81 | 16 | 0 | `grounding_chexagent_better_mass.png` |
| Nodule | 76 | 1 | 0 | `grounding_chexagent_better_nodule.png` |
| *Cardiomegaly (for comparison)* | *141* | *see 5.3: MAIRA-2 better* | | |

What the figures show:
- **Pneumothorax:** in all six examples the pneumothorax is at the top of the patient's left chest.
  CheXagent boxes it there; MAIRA-2 boxes the same region on the patient's right. This repeats the
  side bias seen in the VinDr side test.
- **Mass:** CheXagent's box sits on the mass; MAIRA-2's box is elsewhere, much too small, or
  missing (two of the six had no box).
- **Pneumonia:** in 5 of the 6 examples (13 of all 17 such cases) MAIRA-2 returned no box; it did so
  for 74 of 116 pneumonia cases.
- **Nodule:** only one such case exists; both models rarely box nodules correctly.

![Atelectasis](results/figures/grounding_chexagent_better_atelectasis.png)
![Pleural effusion](results/figures/grounding_chexagent_better_effusion.png)
![Infiltration](results/figures/grounding_chexagent_better_infiltrate.png)
![Pneumonia](results/figures/grounding_chexagent_better_pneumonia.png)
![Pneumothorax](results/figures/grounding_chexagent_better_pneumothorax.png)
![Mass](results/figures/grounding_chexagent_better_mass.png)
![Nodule](results/figures/grounding_chexagent_better_nodule.png)

---

## 6. What the results mean

**1. CheXagent grounds better than MAIRA-2, as pre-registered.** Both hypotheses are supported. The
lead is small at the strict standard (+4.3% at IoU 0.5) and large at the lenient one (+22.2% at IoU
0.3; pointing game 68.5% vs 40.0%).

**2. Neither model draws precise boxes.** At IoU 0.5, only about 1 in 5 radiologist boxes is matched
by either model. CheXagent usually points at the right area (pointing game 68.5%) but draws its box
a different size from the radiologist's. For EDV this matters: the boxes are useful to show
roughly **where** and on **which side**, not to outline a finding precisely.

**3. Cardiomegaly is the exception, in MAIRA-2's favour.** MAIRA-2 matched all 141 heart boxes at IoU
0.5, against 70% for CheXagent. The pooled IoU 0.5 result mixes this with the other conditions.
- *Descriptive, not pre-registered:* without cardiomegaly (806 cases), recall at IoU 0.5 is
  **15.1% for CheXagent vs 4.8% for MAIRA-2**, and at IoU 0.3 it is **44.3% vs 18.1%**.
- So outside the heart, CheXagent's grounding lead is much larger than the pooled +4.3% suggests.

**4. MAIRA-2 often declines to draw a box.** It returned no box in 127 cases (13%), including **74 of
116 pneumonia cases**. There it repeated "Pneumonia" with no coordinates; pneumonia is a diagnosis
rather than a visible shape, which may explain it. CheXagent returned no box in only 9 cases.

**5. How this fits with the earlier results:**

| ability | test | result |
|---|---|---|
| Detection: is the condition there? | AUC, 13 conditions, 3 datasets | CheXagent ahead in 26 of 29 comparisons |
| Side: which side is it on? | VinDr, pre-registered, 153 one-sided findings | CheXagent's side informative (LR+ 6.73); MAIRA-2's not (1.31) |
| **Location: where exactly?** | **this experiment, 947 NIH boxes** | **CheXagent ahead overall; MAIRA-2 ahead on cardiomegaly** |

---

## 7. Limitations

- **One box per finding in NIH**, even when a finding is on both sides. A model that correctly boxes
  both sides can score low (example D). This mostly affects effusion and atelectasis.
- **Box style.** Radiologists' boxes are approximate and follow NIH's own conventions. A model whose
  box style matches them (e.g. MAIRA-2 for the heart) gains, whatever its true accuracy.
- **CheXagent may have seen these images, possibly with these boxes, in training.** NIH is public;
  MAIRA-2 was not trained on it. This could explain part of CheXagent's lead. A second dataset would
  help: VinDr-CXR has boxes from three radiologists, but its images are not square, so CheXagent's
  coordinates must be checked there first.
- **Location only.** Each model was told the condition was present. In real use, a model that
  wrongly thinks a condition is present will also draw a box.
- **One dataset, 8 conditions;** small numbers for some conditions (nodule 76, mass 81). The analysis
  without cardiomegaly was not pre-registered.
- **Earlier exploratory numbers differ slightly.** An earlier check on 416 boxes (3 conditions,
  older MAIRA-2 settings) gave pointing-game rates of 65% vs 33%. The pre-registered numbers here
  replace them.

---

## 8. Suggested summary for the professor

> *"In a pre-registered test on 947 NIH radiologist boxes (8 conditions; each model is given the
> condition name without a side), grounding recall at IoU >= 0.5 was 23.3% for CheXagent vs 19.0% for
> MAIRA-2 (+4.3%, 95% CI +1.7 to +7.0). At IoU >= 0.3 it was 52.5% vs 30.3% (+22.2%, CI +18.8 to
> +25.6). Both models place boxes only roughly; CheXagent finds the right area more often (pointing
> game 68.5% vs 40.0%). MAIRA-2 was better for cardiomegaly (100% vs 70% at IoU 0.5) and returned no
> box in 13% of cases, mostly pneumonia. CheXagent may have seen NIH images in training."*

---

## Files

| What | File |
|---|---|
| Protocol (committed before any box) | `experiments/grounding_recall_protocol.md` |
| Code | `experiments/grounding_recall.py` |
| Results tables | `experiments/results/grounding_recall.md` |
| Every box from each model | `experiments/results/grounding_chexagent.jsonl`, `experiments/results/grounding_maira2.jsonl` |
| Example figure | `experiments/results/figures/grounding_examples.png` |
| Earlier exploratory check | `experiments/results/location_agreement_dev.md` |
| Side test (VinDr, pre-registered) | `experiments/results/vindr_grounding_amendment.md` |
