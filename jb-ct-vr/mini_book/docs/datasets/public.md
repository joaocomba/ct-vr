(public-dataset)=

# Public Dataset
The COVID-CT-MD dataset is composed of 307 labeled CT scans that are used for model training and validation based on a stratified random split: 30% of these CT scans are randomly selected as the validation set, and the remaining are used as the train set. In addition to the train/validation set, the SPGC-ICCASP competition released the SPGC-COVID Test Set (HEIDARIAN et al., 2021), with four independent test sets for models’ evaluation. Three were used to calculate the competition’s results and are applied for performance assessment in our work to enhance comparability with previous approaches. We do not use the fourth Test Set because we would not have fair comparisons concerning the results reported by the competition.


```{figure} /_static/lecture_specific/datasets/table-public.png
---
name: table-public
scale: 50%
---

Public dataset: COVID-CT-MD, number of CT images obtained from each source, and distribution by class.

```

```{figure} /_static/lecture_specific/datasets/public-samples.png
---
name: public-samples
scale: 50%
---

Sample CT slices from the first three test sets. In Test set 1, the noise level is high. In Test 2, some cases reveal cardiovascular-related complications. In Test 3, the image quality and contrast are higher compared to other test sets.

```
