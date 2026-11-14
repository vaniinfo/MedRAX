# EXPLORATORY: CheXagent vs MAIRA-2 in their own words, 20 NIH examples

Images: the 20 examples of `logs/edv_longitudinal_examples_20261020-065035.pdf`. **Selected because CheXagent (two-image score) found a condition MedGemma missed and NIH labels it, so the selection favours CheXagent; examples, not a test.** Same models, prompts and quote-checked reader as `own_words_detection_protocol.md`; current image only; 'stated' = reader answers present (not 'possible'). CheXagent Yes/No = 'Is there X?' with Yes outweighing No (p_present >= 0.5). Ground truth: NIH labels (text-mined).

## Totals over 20 images x 13 conditions (33 labelled present, 227 labelled absent)

| system | caught (of labelled present) | false alarms (of labelled absent) |
|---|---|---|
| CheXagent own report | 19 / 33 | 9 / 227 |
| CheXagent Yes/No | 31 / 33 | 49 / 227 |
| MAIRA-2 own report | 14 / 33 | 12 / 227 |

## Per condition (caught / labelled present; false alarms)

| condition | labelled present | CheXagent own report | CheXagent Yes/No | MAIRA-2 own report |
|---|---|---|---|---|
| cardiomegaly | 1 | 1 / 1; 1 false | 1 / 1; 4 false | 0 / 1; 2 false |
| pleural effusion | 4 | 4 / 4; 4 false | 4 / 4; 4 false | 3 / 4; 4 false |
| pneumothorax | 4 | 3 / 4; 2 false | 4 / 4; 0 false | 1 / 4; 2 false |
| consolidation | 0 | -; 0 false | -; 4 false | -; 0 false |
| pulmonary edema | 1 | 0 / 1; 0 false | 1 / 1; 2 false | 0 / 1; 1 false |
| atelectasis | 7 | 7 / 7; 0 false | 7 / 7; 3 false | 7 / 7; 2 false |
| pneumonia | 1 | 0 / 1; 0 false | 0 / 1; 3 false | 0 / 1; 0 false |
| infiltration | 1 | 0 / 1; 1 false | 0 / 1; 5 false | 0 / 1; 0 false |
| emphysema | 3 | 0 / 3; 0 false | 3 / 3; 1 false | 0 / 3; 0 false |
| nodule | 6 | 1 / 6; 0 false | 6 / 6; 4 false | 1 / 6; 0 false |
| mass | 3 | 2 / 3; 1 false | 3 / 3; 10 false | 1 / 3; 1 false |
| fibrosis | 0 | -; 0 false | -; 2 false | -; 0 false |
| pleural thickening | 2 | 1 / 2; 0 false | 2 / 2; 7 false | 1 / 2; 0 false |

## Per image

Conditions each system states; **bold** = NIH labels it present. NIH labels not caught by any system are listed under 'missed by all'.

| example | image | NIH labels | CheXagent own report | CheXagent Yes/No | MAIRA-2 own report | missed by all |
|---|---|---|---|---|---|---|
| 1 | 00026746_004.png | pneumothorax, nodule, mass | **pneumothorax**, **nodule**, **mass** | **pneumothorax**, emphysema, **nodule**, **mass** | **nodule** | - |
| 2 | 00029511_001.png | pneumothorax, emphysema, nodule | **pneumothorax** | **pneumothorax**, **emphysema**, **nodule**, mass | **pneumothorax** | - |
| 3 | 00008295_018.png | pleural effusion, pleural thickening | **pleural effusion**, **pleural thickening** | **pleural effusion**, consolidation, atelectasis, infiltration, mass, fibrosis, **pleural thickening** | atelectasis, **pleural thickening** | - |
| 4 | 00020751_001.png | atelectasis, emphysema | pneumothorax, **atelectasis** | **atelectasis**, **emphysema**, mass, pleural thickening | **atelectasis** | - |
| 5 | 00021812_001.png | pleural effusion, pneumothorax | **pleural effusion**, **pneumothorax** | **pleural effusion**, **pneumothorax**, consolidation, atelectasis, pneumonia, infiltration, mass, pleural thickening | **pleural effusion** | - |
| 6 | 00000580_008.png | atelectasis, pleural thickening | pleural effusion, **atelectasis** | pleural effusion, **atelectasis**, fibrosis, **pleural thickening** | pleural effusion, **atelectasis** | - |
| 7 | 00003787_002.png | atelectasis | pneumothorax, **atelectasis** | **atelectasis**, nodule, mass | pneumothorax, **atelectasis** | - |
| 8 | 00004031_001.png | nodule, mass | pleural effusion, **mass** | pleural effusion, **nodule**, **mass**, pleural thickening | pleural effusion, **mass** | - |
| 9 | 00005869_001.png | pneumothorax, pneumonia | cardiomegaly, pleural effusion | cardiomegaly, pleural effusion, **pneumothorax**, pulmonary edema, atelectasis, infiltration, nodule, pleural thickening | cardiomegaly, pleural effusion, pulmonary edema | pneumonia |
| 10 | 00006480_002.png | pleural effusion, nodule | **pleural effusion** | **pleural effusion**, **nodule**, pleural thickening | **pleural effusion** | - |
| 11 | 00007065_004.png | nodule | none | **nodule**, mass, pleural thickening | atelectasis | - |
| 12 | 00007422_003.png | pleural effusion, atelectasis | **pleural effusion**, **atelectasis** | **pleural effusion**, **atelectasis**, pleural thickening | **pleural effusion**, **atelectasis** | - |
| 13 | 00007757_007.png | nodule | mass | cardiomegaly, **nodule**, mass | mass | - |
| 14 | 00009488_002.png | atelectasis | **atelectasis** | **atelectasis**, mass | **atelectasis** | - |
| 15 | 00010188_009.png | atelectasis, infiltration | **atelectasis** | cardiomegaly, **atelectasis**, nodule, mass | **atelectasis** | infiltration |
| 16 | 00010627_001.png | pulmonary edema | none | cardiomegaly, consolidation, **pulmonary edema**, pneumonia, infiltration | cardiomegaly | - |
| 17 | 00011121_004.png | atelectasis | pleural effusion, **atelectasis**, infiltration | pleural effusion, consolidation, pulmonary edema, **atelectasis**, pneumonia, infiltration | pleural effusion, **atelectasis** | - |
| 18 | 00013520_010.png | cardiomegaly | **cardiomegaly** | **cardiomegaly** | none | - |
| 19 | 00013904_001.png | mass | none | nodule, **mass** | none | - |
| 20 | 00014725_006.png | emphysema | none | **emphysema**, mass | pneumothorax | - |

## The reports

**Example 1 (00026746_004.png)**

- CheXagent: Lateral **Bilateral hilar and bilateral pulmonary nodules are present.** **There is a small left apical pneumothorax.** **Multiple bilateral pulmonary nodules and masses are present.** The cardiomediastinal silhouette is stable. **There is blunting of the right costophrenic angle, which may represent a small pleural effusion.** A left chest wall Port-A-Cath is present with the tip in the distal SVC.
- MAIRA-2: Port-a-cath through the left subclavian with distal end in the superior vena cava. Multiple bilateral pulmonary nodules in relation to known metastases.

**Example 2 (00029511_001.png)**

- CheXagent: AP The hilar contours are normal. **There is a small right pneumothorax.** The lungs are clear. The cardiomediastinal silhouette is normal. The left costophrenic angle is not included in this radiograph. A right-sided chest tube is present. **Subcutaneous emphysema is present in the right chest wall.**
- MAIRA-2: Right chest tube is present with a small right apical pneumothorax. Patchy and linear opacities in the right mid and lower lung are present, and could be due to atelectasis, aspiration, or infectious pneumonia. Left lung is grossly clear.

**Example 3 (00008295_018.png)**

- CheXagent: Lateral **There is a leftward deviation of the trachea.** <|ref|> Pleural effusion <|/ref|> <|box|> (14,14),(46,72) <|/box|> <|ref|> Pleural thickening <|/ref|> <|box|> (15,14),(46,72) <|/box|> <|ref|> Pleural effusion <|/ref|> <|box|> (15,14),(46,72) <|/box|> <|ref|> Pleural thickening <|/ref|> <|box|> (15,14),(46,72) <|/box|> <|ref|> Pleural thickening <|/ref|> <|box|> (52,14),(74,22) <|/box|> <|ref|> Pleural thickening <|/ref|> <|box|> (52,14),(74,22) <|/box|> The cardiomediastinal silhouette is unchanged. **There is tenting of the left hemidiaphragm.** Surgical clips are present in the right neck.
- MAIRA-2: Post-surgical changes in the right hemithorax with volume loss and diffuse pleural thickening. The left lung is clear.

**Example 4 (00020751_001.png)**

- CheXagent: PA The hilar contours are normal. **There is a small left apical pneumothorax.** **There is volume loss at the left base.** The cardiomediastinal silhouette is normal. **There is persistent elevation of the left hemidiaphragm with left basilar atelectasis.** The left chest tube remains in place.
- MAIRA-2: Left-sided chest tube with end projected over the left lung apex. No significant pneumothorax line is visualized. Left basal subsegmental atelectasis.

**Example 5 (00021812_001.png)**

- CheXagent: LATERAL The hilar contours are normal. **There is a small left apical pneumothorax.** **There is a large left pleural effusion.** The right lung is clear. The heart size is difficult to evaluate. **The left hemidiaphragm is obscured.** The left-sided chest tube is in the same position.
- MAIRA-2: Left-sided pleural drainage catheter has been placed, with apparent resolution of a loculated left pleural effusion. However, a moderate amount of pleural fluid remains, and there is also a probable small basilar pneumothorax.

**Example 6 (00000580_008.png)**

- CheXagent: Lateral **There is persistent prominence of the bilateral hila.** **There is a small right pleural effusion.** **There is associated right basilar atelectasis.** The left lung is clear. There is no pneumothorax. The cardiomediastinal silhouette is unchanged. **There is blunting of the right costophrenic angle, which may represent a small pleural effusion.** A dual lead pacemaker is in place.
- MAIRA-2: Mid-sternotomy cerclage. Pacemaker with bicameral sensors. Right pleural effusion with atelectasis of the underlying lung.

**Example 7 (00003787_002.png)**

- CheXagent: AP The hilar contours are normal. **There is a small right apical pneumothorax.** **There is volume loss at the right base.** The cardiomediastinal silhouette is normal. **There is persistent elevation of the right hemidiaphragm.** A right-sided chest tube is present.
- MAIRA-2: Right chest tube remains in place, with persistent small right apical pneumothorax. Postoperative changes are present in the right hemithorax consistent with recent upper lobe resection. Patchy and linear atelectasis is present in the right mid and both lower lung regions. Subcutaneous emphysema is present in the right chest wall.

**Example 8 (00004031_001.png)**

- CheXagent: Lateral **There is a large right paratracheal mass.** The lungs are clear. There is no pneumothorax. The heart size is normal. **Blunting of the left costophrenic angle is seen.** **The mediastinum is widened, consistent with a known mediastinal mass.**
- MAIRA-2: Left upper lobe mass is again noted. There is a new left pleural effusion. The right lung is clear.

**Example 9 (00005869_001.png)**

- CheXagent: Lateral The hilar contours are normal. **Diffuse interstitial opacities are present.** **There is a small right pleural effusion.** There is no pneumothorax. **The heart is mildly enlarged.** **Blunting of the right costophrenic angle is seen.** The cardiomediastinal silhouette is unremarkable.
- MAIRA-2: Cardiac silhouette is mildly enlarged. Mild pulmonary vascular congestion is present as well as minimal interstitial edema. Small right pleural effusion is also evident. Additional patchy opacities in the mid and lower lungs probably reflect patchy atelectasis, but followup chest radiographs would be helpful to document resolution and to exclude an evolving pneumonia, particularly in the right lower lobe.

**Example 10 (00006480_002.png)**

- CheXagent: PA The hilar contours are normal. The lungs are clear. **There is a small right pleural effusion.** There is no pneumothorax. The heart size is normal. **There is blunting of the right costophrenic angle, consistent with a small right pleural effusion.** The mediastinal contours are normal.
- MAIRA-2: Right costophrenic angle blunting. No other findings.

**Example 11 (00007065_004.png)**

- CheXagent: AP The hilar contours are normal. There is no evidence of acute pneumonia or vascular congestion. There is no pleural effusion. The heart size is normal. The costophrenic angles are sharp. A right-sided central venous catheter is present with the distal lead tip in the distal SVC.
- MAIRA-2: Heart is upper limits of normal in size. Pulmonary vascularity is within normal limits. Right internal jugular vascular catheter terminates in the lower superior vena cava, with no visible pneumothorax. Multifocal patchy and linear opacities are present in the right mid and both lower lung regions, with slight worsening in the right lower lung. These findings favor a multifocal atelectasis, but coexisting aspiration or developing pneumonia is possible in the appropriate clinical setting.

**Example 12 (00007422_003.png)**

- CheXagent: Lateral The hilar contours are normal. **There is a small right pleural effusion.** **There is volume loss at both bases.** The cardiomediastinal silhouette is normal. **There is blunting of the costophrenic angles bilaterally.** The mediastinal contours are normal.
- MAIRA-2: Bibasilar areas of atelectasis are present, with additional small foci of atelectasis in the right mid and left lower lung regions. Small pleural effusions are also evident. Cardiomediastinal contours are within normal limits for technique. Postoperative changes are present in the imaged portion of the upper abdomen, and there is also a small amount of subcutaneous emphysema in the imaged chest wall.

**Example 13 (00007757_007.png)**

- CheXagent: AP **A right suprahilar mass is present.** There is no pneumothorax. **A large right perihilar mass is present.** There is no focal consolidation or pleural effusion. The heart size is normal. The costophrenic angles are clear. The left internal jugular central venous catheter ends in the mid SVC. The mediastinal contours are normal.
- MAIRA-2: PA AP chest demonstrates normal heart size. Left-sided vascular catheter overlies the superior vena cava. There is a 4 cm round radiodensity in the right midlung field.

**Example 14 (00009488_002.png)**

- CheXagent: AP The hilar contours are normal. There is no evidence of pneumothorax. **There is some atelectasis at the lung bases.** The heart size is normal. **There is elevation of the right hemidiaphragm.** A left internal jugular central venous catheter has been placed, terminating in the mid SVC. The mediastinal contours are unremarkable.
- MAIRA-2: Left internal jugular vascular catheter terminates in the proximal superior vena cava, with no visible pneumothorax. Cardiomediastinal contours are within normal limits for technique. Multifocal patchy and linear opacities are present in the right upper and both lower lobes, with slight worsening in the right upper lobe but improvement in the left lower lobe. These findings favor a multifocal atelectasis, but coexisting aspiration or pneumonia is possible in the appropriate clinical setting.

**Example 15 (00010188_009.png)**

- CheXagent: Lateral The hilar contours are normal. **There is a small platelike atelectasis at the left lung base.** There is no evidence of pneumonia. There is no pneumothorax. The cardiomediastinal silhouette is normal. **There is a focal eventration of the right hemidiaphragm.** The left PICC line tip is located in the mid SVC.
- MAIRA-2: Left basal laminar atelectasis. No other alterations.

**Example 16 (00010627_001.png)**

- CheXagent: AP The endotracheal tube is positioned 5 cm above the carina. **There is a slight improvement in the diffuse bilateral pulmonary opacifications.** The cardiomediastinal silhouette is stable. The left costophrenic angle is not included in this study. The left IJ catheter tip is in the lower SVC. The right IJ catheter tip is in the mid SVC.
- MAIRA-2: Tip of endotracheal tube terminates 4.5 cm above the carina, and a nasogastric tube terminates within the stomach. Cardiac silhouette is mildly enlarged accompanied by pulmonary vascular congestion and asymmetrically distributed perihilar and basilar predominant airspace opacities, likely due to pulmonary edema. However, a band-like area of oxygen bronchograms in the right juxtahilar region could reflect a site of focal pulmonary ischemia. Followup radiographs after diuresis may be helpful to exclude other underlying process in this region such as pneumonia or atelectasis.

**Example 17 (00011121_004.png)**

- CheXagent: AP The hilar contours are normal. **There is increased bilateral lower lobe volume loss/infiltrate.** **There is pulmonary vascular redistribution.** The cardiomediastinal silhouette is stable. **There is blunting of the costophrenic angles bilaterally.** The right IJ line tip is in the SVC. The left subclavian line tip is in the SVC.
- MAIRA-2: Right chest tube remains in place, with no visible pneumothorax. Cardiomediastinal contours are stable in appearance. Persistent bibasilar atelectasis and small pleural effusions. Subcutaneous emphysema in the right chest wall is again demonstrated.

**Example 18 (00013520_010.png)**

- CheXagent: AP The endotracheal tube is positioned 5 cm above the carina. There is no evidence of acute pneumonia or vascular congestion. **The cardiac silhouette is mildly enlarged.** The costophrenic angles are clear. The right PICC line tip is in the lower SVC. The NG tube tip is in the stomach.
- MAIRA-2: Tip of endotracheal tube terminates 5.5 cm above the carina, and a nasogastric tube terminates within the stomach. Cardiomediastinal contours are within normal limits for technique. Lungs are grossly clear except for a patchy area of increased opacity in the left retrocardiac region. This may be due to atelectasis, aspiration, and less likely a developing focus of pneumonia. Followup radiographs may be helpful in this regard.

**Example 19 (00013904_001.png)**

- CheXagent: AP **The right hilum is enlarged.** **Innumerable pulmonary metastases are seen.** There is no pneumothorax. The heart size is normal. The left costophrenic angle is not fully included on the image. A left PICC line has been placed, with the tip ending in the right atrium. The right Port-A-Cath is unchanged.
- MAIRA-2: Right internal jugular Port-A-Cath is present, terminating in the lower superior vena cava. Heart size is normal. Widespread pulmonary metastases are present, and have apparently increased in size and number since the recent chest CT. No definite pleural effusion.

**Example 20 (00014725_006.png)**

- CheXagent: AP The hilar contours are normal. There is no evidence of pneumothorax. The cardiomediastinal silhouette is normal. The right hemidiaphragm is mildly elevated. **Subcutaneous emphysema is present in the right chest wall.**
- MAIRA-2: Right chest tube is present, with a small right apical pneumothorax. Patchy and linear opacities in the right mid and lower lung probably represent atelectasis. Subcutaneous emphysema is present in the right chest wall.

