(binary-results)=

# Binary Task

To perform binary classification with our private dataset provided from HCPA and HMV, distinguishing patients classified as having typical COVID-19 lung infection by a radiologist from one of the two hospitals. This binary classification was conceived agglutinating three of the original possible radiological diagnosis given to patients following the RSNA standard; those three classes are negative, atypical, and indeterminate. {numref}`Figure {number} <binary-matrix>` presents our best result for ternary classification using ResNet101 as backbone and Transfer Function 6. For all the binary classification tasks we use COVID-19 label as positive class to obtain all presented metrics. It should be noted that we could not use TheSaviours approach for the lack os annotations at slice-level for our private dataset.


```{figure} /_static/lecture_specific/results/binary-matrix.png
---
name: binary-matrix
scale: 50%
---

Confusion matrix for  binary classification using hospitals (private) dataset.

```


::::{dropdown} **COVID-19 vs. Others Task**

As mentioned COVID-19 is our positive class, while for this experiment for negative class we merged the Negative, Indeterminate, and Atypical classes (from RSNA classification) into a unique **non-COVID-19 class (Others)**. Performance assessment was based on 5-fold CV, using the same folds configuration for COVID-VR, DeCoVNet, and COVNet. The following links present the results obtained by patients for each architecture trained and validated using Transfer Function 6 and ResNet101 as backbone architecture for our COVID-VR approach.
- COVID-VR [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/covid-vs-others/covid-vr.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/covid-vs-others/covid-vr.csv)
- DeCovNet [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/covid-vs-others/decovnet.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/covid-vs-others/decovnet.csv)
- COVNet [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/covid-vs-others/covnet.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/covid-vs-others/covnet.csv)

The {numref}`Figure {number} <table-covid-others>` presents the table which summarizes all the metrics obtained with in this test.

```{figure} /_static/lecture_specific/results/table-private-covid-others.png
---
name: table-covid-others
scale: 50%
---

Comparison of approaches in Covid-19 vs. Others task. Training and validation technique in the private (HMV+HCPA) dataset.

```

The {numref}`Figure {number} <covid-others-comp>` shows the ROC curves for each approach.

```{figure} /_static/lecture_specific/results/roc-private-covid-others.png
---
name: covid-others-comp
scale: 50%
---

ROC curves comparison for binary classification task in private dataset.
```

::::


::::{dropdown} **COVID-19 vs. Normal Task**

In this experiment we consider only the original negative class (i.e., Negative for pneumonia) as the classifiers’ **non-COVID-19 class (Normal)**. Performance assessment was based on 5-fold CV, using the same folds configuration for COVID-VR, DeCoVNet, and COVNet. The following links present the results obtained by patients for each architecture trained and validated using Transfer Function 6 and ResNet101 as backbone architecture for our COVID-VR approach.

- COVID-VR [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/covid-vs-normal/covid-vr.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/covid-vs-normal/covid-vr.csv)
- DeCovNet [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/covid-vs-normal/decovnet.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/covid-vs-normal/decovnet.csv)
- COVNet [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/covid-vs-normal/covnet.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/covid-vs-normal/covnet.csv)

The {numref}`Figure {number} <table-covid-normal>` presents the table which summarizes all the metrics obtained with in this test.

```{figure} /_static/lecture_specific/results/table-private-covid-normal.png
---
name: table-covid-normal
scale: 50%
---

Comparison of approaches in Covid-19 vs. Normal task. Training and validation technique in the private (HMV+HCPA) dataset.

```

The {numref}`Figure {number} <covid-normal-comp>` shows the ROC curves for each approach.

```{figure} /_static/lecture_specific/results/roc-private-covid-normal.png
---
name: covid-normal-comp
scale: 50%
---

ROC curves comparison for binary classification task (Covid-19 vs. Normal) in private dataset.
```

::::


::::{dropdown} **COVID-19 vs. Normal Inference Task**

In this experiment we analyze the generalization of the classification task of COVID-19 against Normal (Negative in RSNA standard), training a model with the preceding detailed parameters (ResNet101 as the backbone and using TF6 to render the segmented lungs).We carry out the training and validation process with data distribution of 90% and 10%, respectively, using only our private dataset for this step. Afterward, we tested over the public dataset (COVID-CT-MD, training and validation set), obtaining an accuracy of 91.5% and 93.5% as F1-measure as shown in table presented in {numref}`Figure {number} <table-inference>`.


- CT-VR [[Table format link]](https://github.com/ct-vr/results/blob/master/binary-task/inference-private-over-public/covid-vr-inference-test.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/binary-task/inference-private-over-public/covid-vr-inference-test.csv)

```{figure} /_static/lecture_specific/results/table-inference.png
---
name: table-inference
scale: 50%
---

Inference test of COVID-19 vs. Normal task classification using Private Dataset for training/validation and the Validation Set of COVID-CT-MD dataset (Public Dataset) to test.


```

::::

