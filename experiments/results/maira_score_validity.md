# EXPLORATORY: does the MAIRA-2 score reflect what MAIRA-2 itself writes? (pleural effusion)

NIH development images, 100 labelled effusion and 100 not (seed 20261023), plus the five worked examples. MAIRA-2 wrote its own findings (bf16, same prompt and image as the score, greedy, 300 tokens). 'States effusion' = a sentence mentions effusion with no negation before it. Not pre-registered; run after the score results were known.

## 1. Faithfulness (no labels used)

- MAIRA-2's own report states effusion in 69 of 200 images.
- AUC of the score for its own statement: 0.807
- Median score when it states effusion: 0.025; when it does not: 0.000

## 2. Detection against the NIH labels

| | NIH effusion (n=100) | NIH no effusion (n=100) |
|---|---|---|
| MAIRA-2's own report states effusion | 65 | 4 |
| CheXagent FOUND (Evidence Engine cut-off) | 68 | 4 |
| MAIRA-2 score AUC (labels) | 0.740 | |
| CheXagent score AUC (labels) | 0.917 | |

## 3. The five worked examples

| image | NIH | CheXagent | MAIRA-2 score | MAIRA-2 report states effusion | MAIRA-2's report |
|---|---|---|---|---|---|
| 00000096_007.png | effusion | 0.948 | 0.015 | yes | The patient is status post median sternotomy and mitral valve replacement. Cardiac silhouette is enlarged, and is accompanied by mild pulmonary vascular congestion. Small pleural effusions are present bilaterally. |
| 00000231_004.png | effusion | 0.965 | 0.047 | yes | The patient is status post recent median sternotomy and coronary artery bypass surgery. Cardiomediastinal contours are within normal limits for post-operative status of the patient. Bibasilar atelectasis is present, worse on the right, and small pleural effusions are also noted. No visible pneumothorax. |
| 00000090_008.png | effusion | 0.555 | 0.001 | no | Changes secondary to right upper lobectomy. Elevation of the right hemidiaphragm. No other findings. |
| 00000006_000.png | no effusion | 0.050 | 0.001 | no | Radiological signs of COPD. |
| 00000029_000.png | no effusion | 0.029 | 0.000 | no | Cardiac silhouette is normal in size. No focal pulmonary opacities, pleural effusions, or pneumothoraces. |
