(ternary-results)=

# Ternary Task

To this task we use the public dataset from SPGC to perform ternary classification using the dataset’s original classes (COVID-19, Normal and CAP) which provided a more direct comparison with other published works that reported their results training and validating with data from the competition. While working with the SPGC dataset, we defined training and validation sets according to the data split published by the ICCASP-SPGC competition. {numref}`Figure {number} <ternary-matrix>` presents our best result for ternary classification using ResNet101 as backbone and Transfer Function 6.


```{figure} /_static/lecture_specific/results/ternary-matrix.png
---
name: ternary-matrix
scale: 50%
---

Confusion matrix for ternary classification task using SPGC (public) dataset.

```


::::{dropdown} **Architecture Comparisons**

The following links present the results obtained by patients for each architecture trained and validated using Transfer Function 6 for all the experiments.
- VGG16 [[Table format link]](https://github.com/ct-vr/results/blob/master/ternary-task/comparing-architectures/vgg16.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/ternary-task/comparing-architectures/vgg16.csv)
- DenseNet121 [[Table format link]](https://github.com/ct-vr/results/blob/master/ternary-task/comparing-architectures/densenet121.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/ternary-task/comparing-architectures/densenet121.csv)
- ResNet101 [[Table format link]](https://github.com/ct-vr/results/blob/master/ternary-task/comparing-architectures/resnet101.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/ternary-task/comparing-architectures/resnet101.csv)
- EfficientNetB0 [[Table format link]](https://github.com/ct-vr/results/blob/master/ternary-task/comparing-architectures/efficientnetb0.csv) [[Raw link]](https://raw.githubusercontent.com/ct-vr/results/master/ternary-task/comparing-architectures/efficientnetb0.csv)
<!-- - VGG19 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-architectures/vgg19.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-architectures/vgg19.csv)
- DenseNet201 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-architectures/densenet201.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-architectures/densenet201.csv)
- EfficientNetB2 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-architectures/efficientnetb2.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-architectures/efficientnetb2.csv) -->

The {numref}`Figure {number} <table-architectures>` summarizes all the metrics obtained with in this test. In overall metrics, ResNet101 gave us better results in terms of performance and calculated metrics.


```{figure} /_static/lecture_specific/results/table-spgc-compare-architectures.png
---
name: table-architectures
scale: 50%
---

Comparison of distinct backbone network architectures in our COVID-VR approach. Model training and validation was carried out with the train and validation sets from the COVID-CT-MD public dataset for the ternary classification task (COVID-19 vs. CAP vs. Normal). F1-score and AUC score are based on the micro-average.

```

The {numref}`Figure {number} <architectures-comp>` shows the ROC curves for one represent of each family architectures.

```{figure} /_static/lecture_specific/results/roc-ternary-spgc-architectures.png
---
name: architectures-comp
scale: 50%
---

Micro-average ROC curves for four distinct Architectures backbones in our COVID-VR approach.
```

::::


::::{dropdown} **Transfer Function Comparisons**

```{figure} /_static/lecture_specific/results/tfs.png
---
name: tfs
scale: 60%
---
Used Transfer Functions for these experiments
```

The following links present the results obtained by patients for each transfer function trained and validated using ResNet101 as backbone.
- Transfer Function 1 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-transfer-functions/tf1.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-transfer-functions/tf1.csv)
- Transfer Function 2 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-transfer-functions/tf2.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-transfer-functions/tf2.csv) 
- Transfer Function 3 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-transfer-functions/tf3.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-transfer-functions/tf3.csv)
- Transfer Function 4 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-transfer-functions/tf4.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-transfer-functions/tf4.csv)
- Transfer Function 5 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-transfer-functions/tf5.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-transfer-functions/tf5.csv)
- Transfer Function 6 [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-transfer-functions/tf6.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-transfer-functions/tf6.csv)

The {numref}`Figure {number} <table-tfs>` summarizes all the metrics obtained with in this test. In overall metrics, Transfer Function 6 gave us better results in terms of performance and calculated metrics.


```{figure} /_static/lecture_specific/results/table-spgc-compare-tfs.png
---
name: table-tfs
scale: 50%
---

Comparison among Transfers Functions in the COVID-VR approach using train and validation sets from the COVID-CT-MD public dataset for the ternary classification task (COVID-19 vs. CAP vs. Normal). F1-score and AUC score are based on the micro-average.
```

The {numref}`Figure {number} <tfs-comp>` shows the ROC curves for one represent of each family architectures.

```{figure} /_static/lecture_specific/results/roc-ternary-spgc-tfs.png
---
name: tfs-comp
scale: 50%
---
Micro-average ROC curves for four distinct Transfer Functions (TF1, TF2, TF3, and TF6) in our COVID-VR approach.
```

::::


::::{dropdown} **Approaches Comparisons**

The following links present the results obtained by patients for each approach trained and validated. Our COVID-VR uses ResNet101 as backbone and Transfer Function.
- COVID-VR [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-approaches/covid-vr.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-approaches/covid-vr.csv)
- DeCovNet [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-approaches/decovnet.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-approaches/decovnet.csv)
- COVNet [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-approaches/covnet.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-approaches/covnet.csv)
- TheSaviours [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/comparing-approaches/the-saviours.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/comparing-approaches/the-saviours.csv)

The {numref}`Figure {number} <table-approaches>` summarizes all the metrics obtained with in this test. In overall metrics, ResNet101 gave us better results in terms of performance and calculated metrics.


```{figure} /_static/lecture_specific/results/table-spgc-compare-approaches.png
---
name: table-approaches
scale: 40%
---

Validation Set − Comparison among COVID-VR and state-of-the-art approaches for ternary classification using the VALIDATION sets from the public dataset provided by the ICCASP-SPGC competition.
```

The {numref}`Figure {number} <approaches-comp>` shows the ROC curves for each approach.

```{figure} /_static/lecture_specific/results/roc-ternary-spgc-approaches.png
---
name: approaches-comp
scale: 50%
---

Micro-average ROC curves for the ternary classification task using the public dataset, considering the Validation set released by the ICCASP-SPGC competition.
```

::::


::::{dropdown} **Test Comparisons (Test set from ICASSP-SPGC 2021)**


The following links present the results obtained by patients for each approach tested in the set used in ICASSP-SPGC Competition 2021 with previously trained approach (for TheSaviours we use the model weights made public by them).
- COVID-VR [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/test-over-icassp-dataset/covid-vr.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/test-over-icassp-dataset/covid-vr.csv)
- DeCovNet [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/test-over-icassp-dataset/decovnet.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/test-over-icassp-dataset/decovnet.csv)
- COVNet [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/test-over-icassp-dataset/covnet.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/test-over-icassp-dataset/covnet.csv)
- TheSaviours [[Table format link]](https://github.com/covid-vr/results/blob/master/ternary-task/test-over-icassp-dataset/the-saviours.csv) [[Raw link]](https://raw.githubusercontent.com/covid-vr/results/master/ternary-task/test-over-icassp-dataset/the-saviours.csv)

The {numref}`Figure {number} <table-approaches-tests>` summarizes all the metrics obtained with in this test. In overall metrics, ResNet101 gave us better results in terms of performance and calculated metrics.

```{figure} /_static/lecture_specific/results/table-spgc-compare-approaches.png
---
name: table-approaches-tests
scale: 40%
---

Test Set − Comparison among COVID-VR and state-of-the-art approaches for ternary classification using the TEST sets from the public dataset provided by the ICCASP-SPGC competition.

```

The {numref}`Figure {number} <test-comp>` shows the ROC curves for each approach.

```{figure} /_static/lecture_specific/results/roc-ternary-spgc-tests.png
---
name: test-comp
scale: 50%
---

Micro-average ROC curves for the ternary classification task using the public dataset, considering the Test set released by the ICCASP-SPGC competition.
```

::::
