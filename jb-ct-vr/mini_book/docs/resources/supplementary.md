# Supplementary Material

Here, we summarize all the links to our repositories and results. Each repository contains an example of how to execute the contained source code.

## Full Pipeline
Our full pipeline presented and detailed in Chapter 5 with its requirements and examples of how to run are available in [[Full Pipeline link]](https://github.com/joaocomba/ct-vr).


### Building
Our project has as main repository [CT VR Docker](https://github.com/joaocomba/ct-vr)
Which depends on 3 repositories included into the project:
- P-HNN Lung Segmentation: A variation of this approach is included in the repository
- Camera Shot Generator[[link]](https://github.com/ct-vr/camera-shots-generator)
- Video Generator[[link]](https://github.com/ct-vr/video-generator), this one is optional (not used for the model, but can be generated while running the pipeline to have access to a video view of the segmented lung with the transfer function)

> [!IMPORTANT]
> You do not need to clone it separately, the `docker-compose build` command clone and install all of them.
> The trained weights of models are lost. You can generate it [using ct-vr-network](https://github.com/joaocomba/ct-vr/tree/main/train-val-network)

Build:

```bash
docker-compose build
```

Full pipeline

``` bash
docker run -it --gpus all --network host --env="DISPLAY" --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,display" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/local/data/:/data/" ct-vr-docker_ctvr:latest python pipeline.py --full_pipeline  --dicom_path /data/dicom/P064
```

Full pipeline with video generator:

```bash
docker run -it --gpus all --network host --env="DISPLAY" --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,display" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/local/data/:/data/" ct-vr-docker_ctvr:latest python pipeline.py --full_pipeline  --dicom_path /data/dicom/P064 --video
```


## Independent Repositories
In this Section we detail the main components of our Full Pipeline which can be used independently. Each repository contains an example of how to execute the code.

### Lung segmentation
Required packages and minimum versions. For complete detail see at require-
ments.txt in our adapted P-HNN version:
- Python 3.6
- Pytorch 1.3
- Cudatoolkit 10.1
- Anaconda

Link to repositories:
- Original P-HNN repository [[link]](https://adampharrison.gitlab.io/p-hnn/).
- Model Weights (Lost link).
<!-- Model Weights [[link]](https://drive.google.com/file/d/1l6yLFScULNw-oVoark0KZ-wnDFX8zwrN/view?usp=sharing).
-->
### Visualization Repositories
Minimum required libraries:
- CMake 3.17
- QT 5.12
- MITK [[link]](https://github.com/MITK/MITK)
- ffmpeg (For video-generator repository)

Link to repositories:
- Repository to capture view images [[link]](https://github.com/ct-vr/camera-shots-generator).
- Repository for video generation from CT Image [[link]](https://github.com/ct-vr/video-generator).

Both repositories need as input an image in NIfTI format and a XML for the transfer functions, samples of both were added to each repository. We highly recommend using a Desktop environment (with UI); however, could be used in a server environment like our Full Pipeline, using VGLRUN command along with the steps described in Full Pipeline link.

### COVID-VR Proposed Network
Required minimum versions (the complete requirements are in the repository link):
- Python 3.6
- TensorFlow 2.0
Link to repositories:
- Repository for train and validation: [[link]](https://github.com/joaocomba/ct-vr/tree/main/train-val-network).
- Model weights (For the two views) (Link Lost).

<!--### Additional Resources
- Repository for get metrics (accuracy, precision, f1-measure, etc) and generate graphics used in this work [[link]](https://github.com/covid-vr/model-evaluation-metrics).
- Repository to generate Grad-CAM visualizations [[link]](https://github.com/covid-vr/covid-vr-grad-cam).
-->