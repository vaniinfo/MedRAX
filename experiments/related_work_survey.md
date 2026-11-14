# Related work survey for the manuscript (2026-10-25)

Four parallel literature searches (web, arXiv and venue pages). [V] = page opened by the searcher;
[S] = seen in search results only, verify before citing. "Not found" means not found in these searches,
not proven absent.

## 1. Image-based, finding-level checking of generated CXR reports (closest threat to novelty)
| Paper | ID | Status | Overlap | Difference from ours |
|---|---|---|---|---|
| Mahmood R. et al., Phrase-grounded fact-checking for automatically generated chest X-ray reports, MICCAI 2025 | arXiv:2509.21356 | V | Same task: per-finding image check incl. location, boxes, ChestX-ray8 and VinDr | Checker trained and given the claim and its location; no laterality-specific test; uncalibrated sigmoid |
| Mahmood R. et al., Anatomically-grounded fact checking of automated CXR reports | arXiv:2412.02177 | V | Detect then LLM-correct | Claim-visible; no calibration |
| Mahmood R. et al., Fine-grained phrasal grounding metric (includes laterality) | arXiv:2412.01031 | V | Laterality-aware grounding of findings | Evaluation metric, claim-visible |
| Zhou N. et al., CoEV counter-evidence verification, MICCAI 2026 | arXiv:2606.18609 | V | Per-claim grounding, masking, correction | Grounder gets the claim; labels from GREEN vs reference |
| Zhang K. et al., RadAgents, MIDL 2026 | arXiv:2509.20490 | V | MAIRA-2 prompted side-free for pneumothorax, side assigned afterwards from lung masks | Not used to audit a report's side; no rationale or measurement of conditioning |
| Heiman A. et al., FactCheXcker, CVPR 2025 | arXiv:2411.18672 | S | Separate image modules correct report claims | Measurements (ETT) only |
| Liao Z. et al., UniVRSE | arXiv:2503.20504 | V | Claim -> VQA check against image | Claim-visible, uncalibrated |
| Hardy R. et al., ReXTrust | arXiv:2412.15264 | V | Finding-level hallucination risk | Uses generator hidden states |
| Zou J. et al., CorBenchX (incl. side-confusion errors) | arXiv:2505.12057 | V/S | Laterality errors in reports | Checker sees the erroneous report |
| Gholipour Picha S. et al., VICCA | arXiv:2501.17726 | V | Grounding DINO verification; one anecdote of a side-named prompt localising that side | Anecdote, not a controlled test |

## 2. Agents with validators, and claim-exposure bias
| Paper | ID | Status | Relevance |
|---|---|---|---|
| Lou J. et al., CXRAgent (EDV) | arXiv:2510.21324 v2 | V | Validator sees the statement; no conditioning study |
| Fallahpour A. et al., MedRAX, ICML 2025 | arXiv:2502.02673 | S | Base framework; grounding prompts carry the hypothesis |
| Jin R. et al., Verification Mirage (medical VQA self-verification agreement bias) | arXiv:2605.10850 | V | Verifiers that see an answer accept wrong ones; mitigated by cross-verification, not blinding |
| Roy R. et al., Source-dependent deference under falsified findings | arXiv:2608.29800 | V | Agents defer to supplied findings |
| Bhowal S. et al., Position, not provenance | arXiv:2607.27304 | S | Laterality edits in VQA reasoning followed least (opposite direction; text answers, not boxes) |
| HEAL-MedVQA / LobA, IJCAI 2025 | arXiv:2505.00744 | S | Swapping anatomy words in CXR VQA questions changes answers |
| Wolf D. et al., Your other Left!, MICCAI 2025 | arXiv:2508.00549 | S | VLMs answer left/right from priors, not the image |
| Hashmi A. et al., AnatomiX | arXiv:2601.03191 | V | Grounding baseline confuses sides on flipped CXRs; CheXagent > MAIRA-2 on detection IoU |
| Yu Z. et al., Counterfactual REC, CVPR 2024 | CVF | S | Natural images: grounders box "cat on the left" when it is on the right |
| Dhuliawala S. et al., Chain-of-Verification (factored variant) | arXiv:2309.11495 | from memory | Text-LLM precedent for answering verification questions without the draft |
| Medical VLM sycophancy: EchoBench 2509.20146; Guo 2509.21979; Aranya 2603.22623 | | S/V | Prompted diagnoses sway medical VLMs |
| CXRAgent-like systems that show the claim to the validator: STRIVE 2608.24237, CogRad 2607.03853, MedAgent-Pro 2503.18968, RadCouncil 2412.06828, X-Ragent 2610.16811, EviAgent 2603.13956 | | S | None blinds its validator |

## 3. Calibrated confidence and CheXagent vs MAIRA-2 evaluations
| Paper | ID | Status | Relevance |
|---|---|---|---|
| Elyassirad D. et al., CONRep (MedGemma first-token Yes/No + conformal), 2026 | arXiv:2602.03910 | V | Closest to p_present calibration; conformal sets, not LRs |
| Khanmohammadi R. et al., Calibrated triage, not autonomy | arXiv:2606.15910 | V | Confidence estimators for medical VLMs; high-confidence error region matters |
| Byun J.Y., Overconfidence in medical VQA | arXiv:2604.02543 | V | Platt fixes calibration, not discrimination |
| Gao Y. et al., Next-word probability is not pre-test probability | arXiv:2411.04962 | V | Motivates calibrating token probabilities |
| Akhtar M. et al., Auditing medical VLMs for TB screening | arXiv:2609.21763 | V | Frozen thresholds often fail to transfer |
| Lotfinia M. et al., CXR VLMs do not always need the image | arXiv:2606.17710 | V | Reviewers will ask for image-ablation controls |
| Bani-Harouni D. et al., ConRad (GRPO for calibrated verbal confidence, MedGemma) | arXiv:2603.29492 | V | Generator self-confidence, not independent checker |
| RadVLM (Deperrois N.) 2502.03333; CheXGround 2608.30758; CheXOne 2604.00493 | | V | Grounding head-to-heads incl. CheXagent and MAIRA-2 (not on NIH boxes) |
| cxr-grounding-benchmark (GitHub, anatomy queries) | | V | CheXagent ranks last on anatomy grounding: a contradicting ranking to pre-empt |

## 4. Longitudinal report generation and RL
| Paper | ID | Status | Relevance |
|---|---|---|---|
| Bannur S. et al., MAIRA-2 | arXiv:2406.04449 | V | Already uses prior image + prior report + indication; key baseline |
| Zhang et al., EditGRPO, AACL 2025 | arXiv:2509.22812 | V | SFT+GRPO with prior images, OOD tests; excludes prior reports to avoid copying |
| Maeng J. et al., STRIVE | arXiv:2608.24237 | V | Progression-aware GRPO; LCC change-correctness metric |
| Zhang Y. et al., CheXOne | arXiv:2604.00493 | V | SFT+GRPO, prior-aware progression description |
| Sellergren A. et al., MedGemma 1.5 technical report | arXiv:2604.05081 | V | Longitudinal only as classification; temporal mentions removed from CXR training reports (explains base 3/499) |
| Berger et al., ChexReason: RL optimises benchmarks, not patients | arXiv:2512.23090 | V | GRPO after SFT reduced transfer to NIH: needs SFT-only ablation |
| Banerjee O. et al., DPO against hallucinated prior exams | arXiv:2406.06496 | V | Comparison language can be hallucinated: needs no-prior / shuffled-prior controls |
| CT-ΔBench (MedGemma 1.5 4B LoRA for CT difference reports) | arXiv:2608.11534 | S | Closest MedGemma-1.5 comparison-report precedent |
| Zhu Q. et al., Longitudinal-MIMIC, MICCAI 2023 | arXiv:2306.08749 | S | Prior report as input; standard benchmark |
| Libra 2411.19378; HC-LLM 2412.11070; MLRG 2502.20056; CXRMate 2307.09758 | | V | Prior-aware baselines |
| Change-correctness evaluation: MS-CXR-T (BioViL-T 2301.04558), L-MIMIC annotator 2601.16753, LUNGUAGE 2505.21190 | | V/S | What reviewers will expect |

## 5. Reporting guidelines
STARD-AI (Nat Med 2025) as primary; CLAIM 2024; TRIPOD-LLM (Nat Med 2025) for prompts and decoding;
NeurIPS pre-registration workshops and Hofman 2023 (arXiv:2311.18807) for the pre-registration design.
