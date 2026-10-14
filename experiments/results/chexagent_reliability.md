# Where is CheXagent right? Reliability of its decisions

Decision as in the review tool: FOUND = support cut-off reached; AGAINST = strong-no; otherwise uncertain. Engine v1.0 cut-offs for the six, provisional NIH cut-offs for the seven. Wilson 95% intervals. **NIH dev** is the data the cut-offs were fitted on (in-sample); **NIH test** and **VinDr** are held out; Indiana was discovery data; the MIMIC pilot uses preliminary automatic labels. Precision and NPV depend on how common the condition is in each dataset; the likelihood ratios do not.

| condition | dataset | positives / n | FOUND: right (precision) | FOUND: share of positives caught | LR+ | AGAINST: right (NPV) | AGAINST: positives wrongly ruled out | AGAINST made on |
|---|---|---|---|---|---|---|---|---|
| cardiomegaly | Indiana | 164 / 544 | 83% (104/126) [75%-88%] | 63% (104/164) [56%-70%] | 11.0 | 100% (161/161) [98%-100%] | 0 of 164 | 30% of images |
| cardiomegaly | NIH dev | 80 / 3500 | 25% (47/186) [20%-32%] | 59% (47/80) [48%-69%] | 14.5 | 100% (2061/2061) [100%-100%] | 0 of 80 | 59% of images |
| cardiomegaly | NIH test | 35 / 1500 | 26% (18/68) [17%-38%] | 51% (18/35) [36%-67%] | 15.1 | 100% (876/877) [99%-100%] | 1 of 35 | 58% of images |
| cardiomegaly | VinDr | 358 / 3000 | 74% (314/424) [70%-78%] | 88% (314/358) [84%-91%] | 21.1 | 100% (1434/1434) [100%-100%] | 0 of 358 | 48% of images |
| cardiomegaly | MIMIC pilot (19) | 7 / 19 | 67% (4/6) [30%-90%] | 57% (4/7) [25%-84%] | 3.4 | 100% (2/2) [34%-100%] | 0 of 7 | 11% of images |
| pleural effusion | Indiana | 120 / 544 | 84% (89/106) [76%-90%] | 74% (89/120) [66%-81%] | 18.5 | 100% (136/136) [97%-100%] | 0 of 120 | 25% of images |
| pleural effusion | NIH dev | 180 / 3500 | 47% (124/264) [41%-53%] | 69% (124/180) [62%-75%] | 16.3 | 100% (1871/1879) [99%-100%] | 8 of 180 | 54% of images |
| pleural effusion | NIH test | 77 / 1500 | 44% (50/113) [35%-53%] | 65% (50/77) [54%-75%] | 14.7 | 99% (780/788) [98%-99%] | 8 of 77 | 53% of images |
| pleural effusion | VinDr | 120 / 3000 | 83% (97/117) [75%-89%] | 81% (97/120) [73%-87%] | 116.4 | 100% (2224/2224) [100%-100%] | 0 of 120 | 74% of images |
| pleural effusion | MIMIC pilot (19) | 6 / 19 | 100% (5/5) [57%-100%] | 83% (5/6) [44%-97%] | inf | 100% (3/3) [44%-100%] | 0 of 6 | 16% of images |
| pneumothorax | Indiana | 28 / 544 | 83% (19/23) [63%-93%] | 68% (19/28) [49%-82%] | 87.5 | 100% (32/32) [89%-100%] | 0 of 28 | 6% of images |
| pneumothorax | NIH dev | 43 / 3500 | 24% (21/89) [16%-33%] | 49% (21/43) [35%-63%] | 24.8 | 100% (631/633) [99%-100%] | 2 of 43 | 18% of images |
| pneumothorax | NIH test | 18 / 1500 | 26% (12/46) [16%-40%] | 67% (12/18) [44%-84%] | 29.1 | 100% (248/249) [98%-100%] | 1 of 18 | 17% of images |
| pneumothorax | VinDr | 15 / 3000 | 29% (15/52) [18%-42%] | 100% (15/15) [80%-100%] | 80.7 | 100% (1482/1482) [100%-100%] | 0 of 15 | 49% of images |
| pneumothorax | MIMIC pilot (19) | 1 / 19 | 50% (1/2) [9%-91%] | 100% (1/1) [21%-100%] | 18.0 | 100% (1/1) [21%-100%] | 0 of 1 | 5% of images |
| consolidation | Indiana | 109 / 544 | 75% (15/20) [53%-89%] | 14% (15/109) [9%-21%] | 12.0 | n/a | 0 of 109 | 0% of images |
| consolidation | NIH dev | 62 / 3500 | 13% (6/45) [6%-26%] | 10% (6/62) [5%-20%] | 8.5 | n/a | 0 of 62 | 0% of images |
| consolidation | NIH test | 27 / 1500 | 22% (4/18) [9%-45%] | 15% (4/27) [6%-32%] | 15.6 | n/a | 0 of 27 | 0% of images |
| consolidation | VinDr | 25 / 3000 | 35% (18/52) [23%-48%] | 72% (18/25) [52%-86%] | 63.0 | n/a | 0 of 25 | 0% of images |
| consolidation | MIMIC pilot (19) | 0 / 19 | n/a | n/a | n/a | n/a | 0 of 0 | 0% of images |
| pulmonary edema | Indiana | 45 / 544 | 44% (31/70) [33%-56%] | 69% (31/45) [54%-80%] | 8.8 | 96% (23/24) [80%-99%] | 1 of 45 | 4% of images |
| pulmonary edema | NIH dev | 24 / 3500 | 12% (14/121) [7%-18%] | 58% (14/24) [39%-76%] | 19.0 | 100% (667/668) [99%-100%] | 1 of 24 | 19% of images |
| pulmonary edema | NIH test | 10 / 1500 | 16% (7/43) [8%-30%] | 70% (7/10) [40%-89%] | 29.0 | 100% (286/286) [99%-100%] | 0 of 10 | 19% of images |
| pulmonary edema | MIMIC pilot (19) | 0 / 19 | 0% (0/4) [0%-49%] | n/a | n/a | n/a | 0 of 0 | 0% of images |
| atelectasis | Indiana | 161 / 544 | 64% (56/87) [54%-74%] | 35% (56/161) [28%-42%] | 4.3 | 100% (54/54) [93%-100%] | 0 of 161 | 10% of images |
| atelectasis | NIH dev | 218 / 3500 | 45% (94/209) [38%-52%] | 43% (94/218) [37%-50%] | 12.3 | 100% (997/1000) [99%-100%] | 3 of 218 | 29% of images |
| atelectasis | NIH test | 93 / 1500 | 43% (45/104) [34%-53%] | 48% (45/93) [38%-58%] | 11.5 | 100% (409/410) [99%-100%] | 1 of 93 | 27% of images |
| atelectasis | VinDr | 10 / 3000 | 9% (9/104) [5%-16%] | 90% (9/10) [60%-98%] | 28.3 | 100% (1229/1229) [100%-100%] | 0 of 10 | 41% of images |
| atelectasis | MIMIC pilot (19) | 8 / 19 | 60% (3/5) [23%-88%] | 38% (3/8) [14%-69%] | 2.1 | n/a | 0 of 8 | 0% of images |
| emphysema | NIH dev | 44 / 3500 | 24% (19/78) [16%-35%] | 43% (19/44) [30%-58%] | 25.3 | n/a | 0 of 44 | 0% of images |
| emphysema | NIH test | 14 / 1500 | 24% (6/25) [11%-43%] | 43% (6/14) [21%-67%] | 33.5 | n/a | 0 of 14 | 0% of images |
| emphysema | MIMIC pilot (19) | 4 / 19 | 43% (3/7) [16%-75%] | 75% (3/4) [30%-95%] | 2.8 | n/a | 0 of 4 | 0% of images |
| nodule | NIH dev | 155 / 3500 | 36% (58/162) [29%-43%] | 37% (58/155) [30%-45%] | 12.0 | 100% (456/456) [99%-100%] | 0 of 155 | 13% of images |
| nodule | NIH test | 85 / 1500 | 28% (22/79) [19%-39%] | 26% (22/85) [18%-36%] | 6.4 | 99% (180/181) [97%-100%] | 1 of 85 | 12% of images |
| nodule | MIMIC pilot (19) | 2 / 19 | 100% (1/1) [21%-100%] | 50% (1/2) [9%-91%] | inf | n/a | 0 of 2 | 0% of images |
| mass | NIH dev | 123 / 3500 | 40% (58/144) [33%-48%] | 47% (58/123) [39%-56%] | 18.5 | n/a | 0 of 123 | 0% of images |
| mass | NIH test | 67 / 1500 | 47% (35/74) [36%-59%] | 52% (35/67) [40%-64%] | 19.2 | n/a | 0 of 67 | 0% of images |
| mass | MIMIC pilot (19) | 0 / 19 | n/a | n/a | n/a | n/a | 0 of 0 | 0% of images |
| fibrosis | NIH dev | 56 / 3500 | 13% (13/100) [8%-21%] | 23% (13/56) [14%-36%] | 9.2 | n/a | 0 of 56 | 0% of images |
| fibrosis | NIH test | 24 / 1500 | 9% (3/34) [3%-23%] | 12% (3/24) [4%-31%] | 6.0 | n/a | 0 of 24 | 0% of images |
| fibrosis | MIMIC pilot (19) | 1 / 19 | n/a | 0% (0/1) [0%-79%] | n/a | n/a | 0 of 1 | 0% of images |
| pleural thickening | NIH dev | 80 / 3500 | 19% (15/77) [12%-30%] | 19% (15/80) [12%-29%] | 10.3 | n/a | 0 of 80 | 0% of images |
| pleural thickening | NIH test | 30 / 1500 | 25% (9/36) [14%-41%] | 30% (9/30) [17%-48%] | 16.3 | n/a | 0 of 30 | 0% of images |
| pleural thickening | MIMIC pilot (19) | 2 / 19 | 25% (1/4) [5%-70%] | 50% (1/2) [9%-91%] | 2.8 | n/a | 0 of 2 | 0% of images |
| pneumonia | NIH dev | 28 / 3500 | n/a | 0% (0/28) [0%-12%] | n/a | n/a | 0 of 28 | 0% of images |
| pneumonia | NIH test | 9 / 1500 | n/a | 0% (0/9) [0%-30%] | n/a | n/a | 0 of 9 | 0% of images |
| pneumonia | MIMIC pilot (19) | 1 / 19 | n/a | 0% (0/1) [0%-79%] | n/a | n/a | 0 of 1 | 0% of images |
| infiltration | NIH dev | 430 / 3500 | n/a | 0% (0/430) [0%-1%] | n/a | n/a | 0 of 430 | 0% of images |
| infiltration | NIH test | 185 / 1500 | n/a | 0% (0/185) [0%-2%] | n/a | n/a | 0 of 185 | 0% of images |
| infiltration | MIMIC pilot (19) | 0 / 19 | n/a | n/a | n/a | n/a | 0 of 0 | 0% of images |
