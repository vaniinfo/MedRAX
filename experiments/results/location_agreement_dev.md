# EXPLORATORY: location agreement as evidence (development data)

Indiana: development split, side/zone from MeSH qualifiers. NIH: radiologist boxes (BBox_List_2017.csv), frozen test-split images excluded. 'Agrees with a claim' = the model boxes the claimed side. Wrong-side claims are the true side flipped (unilateral findings only). LR = P(agree | correct side) / P(agree | wrong side) with a +0.5 correction, 95% bootstrap CI. Caution: NIH usually has one radiologist box per finding even when the finding is bilateral, so for NIH effusion and atelectasis a model that boxes both sides is counted as agreeing with the 'wrong' side; NIH pneumothorax is not affected in this way.

## CheXagent

| data | finding | true cases | boxed | side exactly right | unilateral: true side boxed | unilateral: wrong side only | agree, correct claim | agree, wrong claim | LR for side | absent findings: agree with random claim |
|---|---|---|---|---|---|---|---|---|---|---|
| indiana | pleural effusion | 79 | 100% (79/79) | 77% (61/79) | 100% (38/38) | 0% (0/38) | 100% (38/38) | 24% (9/38) | 4.1 (2.7-8.6) | 40% (16/40) |
| indiana | atelectasis | 100 | 98% (98/100) | 58% (58/100) | 94% (59/63) | 3% (2/63) | 94% (59/63) | 17% (11/63) | 5.2 (3.3-9.6) | 22% (9/40) |
| indiana | consolidation | 67 | 93% (62/67) | 60% (40/67) | 89% (39/44) | 5% (2/44) | 89% (39/44) | 5% (2/44) | 15.8 (7.0-83.0) | 45% (18/40) |
| indiana | pneumothorax | 14 | 93% (13/14) | 93% (13/14) | 93% (13/14) | 0% (0/14) | 93% (13/14) | 0% (0/14) | 27.0 (23.0-29.0) | 12% (5/40) |
| nih | pleural effusion | 150 | 99% (149/150) | 46% (69/150) | 97% (146/150) | 2% (3/150) | 97% (146/150) | 53% (80/150) | 1.8 (1.6-2.2) | - |
| nih | atelectasis | 173 | 99% (171/173) | 71% (122/173) | 87% (151/173) | 12% (20/173) | 87% (151/173) | 28% (49/173) | 3.1 (2.5-4.0) | - |
| nih | pneumothorax | 93 | 100% (93/93) | 95% (88/93) | 96% (89/93) | 4% (4/93) | 96% (89/93) | 5% (5/93) | 16.3 (8.9-59.0) | - |

- Indiana zone (a box in the true zone, where MeSH gives one): 40% (66/165)
- NIH pointing game (a box centre inside a radiologist box): 65% (269/416)

## MAIRA-2

| data | finding | true cases | boxed | side exactly right | unilateral: true side boxed | unilateral: wrong side only | agree, correct claim | agree, wrong claim | LR for side | absent findings: agree with random claim |
|---|---|---|---|---|---|---|---|---|---|---|
| indiana | pleural effusion | 79 | 100% (79/79) | 25% (20/79) | 58% (22/38) | 42% (16/38) | 58% (22/38) | 47% (18/38) | 1.2 (0.8-1.9) | 48% (19/40) |
| indiana | atelectasis | 100 | 95% (95/100) | 33% (33/100) | 54% (34/63) | 41% (26/63) | 54% (34/63) | 46% (29/63) | 1.2 (0.8-1.7) | 30% (12/40) |
| indiana | consolidation | 67 | 84% (56/67) | 27% (18/67) | 45% (20/44) | 36% (16/44) | 45% (20/44) | 50% (22/44) | 0.9 (0.6-1.4) | 45% (18/40) |
| indiana | pneumothorax | 14 | 100% (14/14) | 71% (10/14) | 79% (11/14) | 21% (3/14) | 79% (11/14) | 29% (4/14) | 2.6 (1.4-8.3) | 40% (16/40) |
| nih | pleural effusion | 150 | 99% (149/150) | 47% (71/150) | 57% (86/150) | 42% (63/150) | 57% (86/150) | 52% (78/150) | 1.1 (0.9-1.4) | - |
| nih | atelectasis | 173 | 92% (159/173) | 47% (82/173) | 57% (98/173) | 35% (61/173) | 57% (98/173) | 45% (77/173) | 1.3 (1.0-1.6) | - |
| nih | pneumothorax | 93 | 100% (93/93) | 54% (50/93) | 65% (60/93) | 35% (33/93) | 65% (60/93) | 46% (43/93) | 1.4 (1.1-1.8) | - |

- Indiana zone (a box in the true zone, where MeSH gives one): 32% (53/165)
- NIH pointing game (a box centre inside a radiologist box): 33% (136/416)

## Both models

Unilateral true findings: claim agreement when both models box the claimed side.

| data | agree (both), correct claim | agree (both), wrong claim | the two models' sides match |
|---|---|---|---|
| indiana | 51% (81/159) | 8% (12/159) | 39% (101/260) |
| nih | 55% (227/416) | 17% (72/416) | 40% (168/416) |

See location_side_phrase_test.md for the side-in-phrase test.
