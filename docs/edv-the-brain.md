# The Brain Behind EDV

What Evidence-Driven Validation actually knows, in plain language — every model, every
finding, and the measurements that make the system's confidence mean something.

Source material for slides. Every number here is generated from committed data; see
[Slide 13](#slide-13--where-the-numbers-come-from).

**How to use this file.** One section per slide, numbered. Each carries a draft
**Frame** line — the single point that slide should land. Tell me the slide number and
how you want it framed, and I will rewrite that section; the numbering stays stable so
references keep working.

## Slide index

| # | slide | one line |
|---|---|---|
| 1 | [The one idea](#slide-1--the-one-idea) | an answer is not evidence |
| 2 | [The cast](#slide-2--the-cast) | five models, and who grades nothing |
| 3 | [The two questions](#slide-3--the-two-questions-asked-of-every-model) | yes and no are measured separately |
| 4 | [When it says YES](#slide-4--when-it-says-yes) | the best cell is 74% |
| 5 | [When it says NO](#slide-5--when-it-says-no) | every number looks superb |
| 6 | [The same table, honestly scored](#slide-6--the-same-table-honestly-scored) | most of that was free credit |
| 7 | [No model wins everywhere](#slide-7--no-model-wins-everywhere) | why reliability is keyed three ways |
| 8 | [Agreement is often worthless](#slide-8--agreement-is-often-worthless) | they make the same mistakes |
| 9 | [How the evidence combines](#slide-9--how-the-evidence-combines) | anchor and discount, not voting |
| 10 | [Can a model point at it?](#slide-10--can-a-model-point-at-what-it-found) | a box is not a second opinion |
| 11 | [What it is worth](#slide-11--what-the-finished-system-is-worth) | 77.7%, so the top grade is Medium |
| 12 | [Honest limits](#slide-12--honest-limits) | one hospital, thin pneumothorax, floors |
| 13 | [Where the numbers come from](#slide-13--where-the-numbers-come-from) | backup / provenance |

---

## Slide 1 — The one idea

**Frame:** everything else in the deck is this sentence with numbers filled in.

> A model's answer is not evidence. **What that model has been measured to be right
> about, when it answers that way, about that specific finding** — that is evidence.

Conventional agent design asks a language model to judge how good the evidence is. EDV
refuses to. The evidence is scored arithmetically from measurements taken beforehand, and
the language model is told the result.

---

## Slide 2 — The cast

**Frame:** four models produce evidence; the famous one is not allowed to grade it.

| | what it is | what it produces |
|---|---|---|
| **CheXagent** | 3B radiology vision-language model (Stanford) | a probability, 0–1 |
| **DenseNet** | classic image classifier (TorchXRayVision) | a probability, 0–1 |
| **MedGemma** | 4B medical vision-language model (Google) | effectively just Yes or No |
| **Report generator** | writes a radiology report (2 models) | prose — scored on what it asserts |
| **GPT-4o** | the Director | picks tools, writes the answer. **Grades nothing.** |

The six findings: cardiomegaly, pleural effusion, pneumothorax, consolidation, pulmonary
edema, atelectasis.

---

## Slide 3 — The two questions asked of every model

**Frame:** "is this model good?" is the wrong question, and asking it is where confidence
goes wrong.

Instead, for every model and every finding:

- when it says **yes**, how often is it right?
- when it says **no**, how often is it right?

Those turn out to be wildly different numbers. That is why they are measured separately —
and the next three slides are what that separation reveals.

---

## Slide 4 — When it says YES

**Frame:** no single tool's positive claim is trustworthy. Not one.

*The finding is present — how often is that true?*

| finding | how common | CheXagent | DenseNet | MedGemma | Report |
|---|---|---|---|---|---|
| cardiomegaly | 29% | 62% | 53% | **68%** | 65% |
| pleural effusion | 22% | **74%** | 50% | 70% | 69% |
| pneumothorax | 5% | 40% | **8%** | 55% | 46% |
| consolidation | 20% | 44% | 34% | 48% | 56% |
| pulmonary edema | 8% | 46% | 21% | **60%** | 27% |
| atelectasis | 30% | 46% | 41% | 44% | 45% |

The best cell in the table is 74%. Most are coin flips. A positive claim from any single
tool is a lead, not a conclusion.

---

## Slide 5 — When it says NO

**Frame:** set this up as good news, because the next slide takes it away.

*The finding is absent — how often is that true?*

| finding | CheXagent | DenseNet | MedGemma |
|---|---|---|---|
| cardiomegaly | 97% | 94% | 92% |
| pleural effusion | 95% | 96% | 92% |
| pneumothorax | 99% | 96% | 96% |
| consolidation | 91% | 92% | 88% |
| pulmonary edema | 98% | 97% | 96% |
| atelectasis | 94% | 86% | 88% |

Every number is high. Every model looks excellent at ruling things out.

---

## Slide 6 — The same table, honestly scored

**Frame:** most of that was free credit. This is the trap the whole system exists to avoid.

Pneumothorax occurs in 5% of films. A model that answers "no" to every single film is
right **94.9%** of the time while knowing nothing. MedGemma's 96% beats a rock by one
point.

So each score has the free credit subtracted, giving a **strength**:

| finding | CheXagent | DenseNet | MedGemma |
|---|---|---|---|
| cardiomegaly | 0.89 | 0.80 | 0.72 |
| pleural effusion | 0.77 | 0.83 | 0.64 |
| pneumothorax | **0.76** | 0.31 | **0.20** |
| consolidation | 0.57 | 0.61 | 0.41 |
| pulmonary edema | 0.74 | 0.68 | 0.45 |
| atelectasis | 0.79 | 0.55 | 0.60 |

MedGemma's pneumothorax "no" collapses from a flattering 96% to 0.20. CheXagent's holds
at 0.76. Identical-looking answers, worth roughly four times as much from one model as
from the other.

Taking slides 4–6 together: **ruling out works, ruling in does not.** When this system
says something is absent, that is its strongest output.

---

## Slide 7 — No model wins everywhere

**Frame:** this is why a single trust score per model would have been wrong in both
directions at once.

CheXagent is best overall, but **MedGemma beats it** on cardiomegaly and edema positives.
DenseNet is competitive on effusion and **near-useless on pneumothorax** — 8%.

So reliability is stored per **tool × finding × direction**. Three keys, not one.

---

## Slide 8 — Agreement is often worthless

**Frame:** three tools agreeing can be one mistake copied three times.

Because they make *the same mistakes*:

| finding | how correlated their errors are | so agreement is… |
|---|---|---|
| pneumothorax | 0.12 – 0.17 | **genuinely meaningful** |
| pulmonary edema | 0.23 – 0.46 | moderately meaningful |
| cardiomegaly | 0.46 – 0.55 | partly redundant |
| consolidation | 0.38 – 0.58 | partly redundant |
| **atelectasis** | **0.60 – 0.64** | **nearly worthless** |

Real case: all three tools called atelectasis on film `3453_IM-1676`, labelled normal.

---

## Slide 9 — How the evidence combines

**Frame:** not voting. Counting agreeing tools is exactly the error slide 8 exposes.

The strongest single piece of evidence **anchors** the conclusion. Each additional
agreeing tool adds something, **discounted by how correlated its errors are** with what
has already been counted.

On pneumothorax a second tool adds a lot. On atelectasis it adds almost nothing.

That yields a **net score**. One boundary, **0.48**, survived testing on held-out data.

---

## Slide 10 — Can a model point at what it found?

**Frame:** a box looks like confirmation and mostly is not — and we measured that rather
than assuming it.

CheXagent can draw a box around a named finding. On 573 films it boxes 96% of positives
and 49% of negatives — better than MAIRA-2, which boxes a pneumothorax on films that have
none.

But the honest question is what a box adds *given the same model already answered*:

| that model's own yes/no answer | drew a box? | actually had the finding |
|---|---|---|
| **yes** | boxed 256 of 257 times | — no information at all |
| **no** | box drawn | 37% |
| **no** | no box | 7% |

When it has already said yes, the box is automatic. When it has said no, a box is real but
weak — and 37% still means more likely absent than present.

Grounding is never counted as a second agreeing tool. It says *where* a finding would be,
not *that* it is there.

---

## Slide 11 — What the finished system is worth

**Frame:** the result is real, and it is deliberately not called High.

Tested on 272 films it had never seen, every number frozen beforehand:

| | positive claims correct |
|---|---|
| net score **above 0.48** | **77.7%** |
| net score **below 0.48** | **56.9%** |

The confidence intervals do not overlap, so the boundary is real.

But 77.7% is not what "High confidence" should mean. So **the top grade is Medium, and
there is no High.** No evidence supported a second boundary, so no third tier was
invented.

---

## Slide 12 — Honest limits

**Frame:** say these before someone else does.

- **One hospital.** All Indiana University films. Internal validation, not external.
- **Pneumothorax is thin.** 28 positive cases — over the minimum of 20, but barely.
  Nothing in the calibration is validated for it.
- **Labels are indexing, not re-reads.** Ground truth comes from curated MeSH headings,
  and absence of a heading is treated as absence of the finding. That indexing is not
  exhaustive, so every measured number here is a **floor**, not a ceiling.

---

## Slide 13 — Where the numbers come from

**Frame:** backup slide. Pull it up only if someone asks whether this is reproducible.

| table | lives in | rows |
|---|---|---|
| `RELIABILITY` | `medrax/agent/validator.py` | 18 — 3 tools × 6 findings |
| `DEPENDENCE` | `medrax/agent/validator.py` | 18 — pairwise error correlation |
| `REPORT_RELIABILITY` | `medrax/agent/validator.py` | 6 — report generator text |
| `GROUNDING_MEASURED` | `medrax/agent/validator.py` | the box experiment |

None are hand-written. The pipeline is:

```
scripts/build_eval_set.py       select the films (reproducible from --seed)
scripts/measure_reliability.py  run every tool over them
scripts/analyze_reliability.py  bootstrap intervals, and refuse underpowered rows
```

`analyze_reliability.py` enforces `MIN_POSITIVES = 20`: a finding with fewer positive
cases is noise, not a measurement, and never reaches the table.

Per-film readings are committed, so all of this is re-derivable without a GPU:

| file | contents |
|---|---|
| `reliability.json` | 544 films — the measurement corpus |
| `heldout.json` | 272 films — the frozen calibration set |
| `ablation.json` | 900 claims — Director alone vs Director + EDV |
| `grounding_specificity.json` | 573 pairs — the box experiment |
