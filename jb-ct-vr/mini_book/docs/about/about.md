# CT-VR

Accurate classification of pulmonary diseases is critical for clinical decision-making, and deep learning models using chest CT scans have become a key tool in this task. Most existing approaches rely on 2D CT slices, which provide limited views and may miss important spatial patterns across the lung volume. To address this, we introduce CT-VR, a novel classification approach that leverages 3D volume-rendered images captured from multiple angles. By incorporating multi-view volume rendering, CT-VR enhances the model ability to detect and differentiate between pulmonary conditions. We evaluate the method using COVID-19 datasets as a primary case study, which include private datasets from partner hospitals and a publicly available benchmark. Results demonstrate that our approach improves lesion identification and delivers performance compared to traditional slice-based models, highlighting its potential as a more effective solution for lung disease classification.

```{figure} /_static/lecture_specific/index/pipeline.png
---
name: pipeline
scale: 50%
---

**Figure 1:** The pipeline of our proposed CT-VR approach is divided into two stages. Stage 1 carries out data preparation to obtain the input images for model development, which is conducted in Stage 2. Data preparation involves the steps of image resizing by interpolation, lung segmentation, 3D volume rendering using pre-defined transfer functions, and snapshots generation for axial and coronal planes In Stage 2, DL models are trained for each plane to distinguish among the classes of interest and their outputs are combined to obtain a final patient-level classification.

```
