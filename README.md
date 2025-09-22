# A Deep Learning Framework for Pulmonary Disease Classification Using Volume-Rendered CTs (2025)

[![CI](https://github.com/<you>/pulmonary-dl-ct-2025/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/pulmonary-dl-ct-2025/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](https://www.python.org/)

Reproducible artifact for the paper **A Deep Learning Framework for Pulmonary Disease Classification Using Volume-Rendered CTs**.

**Authors**: Noemi Maritza L. Romero, Ricco V. C. Soares, Mariana Recamonde-Mendoza, João L. D. Comba  
**Affiliation**: Instituto de Informática, UFRGS

## Quick Start

> [!IMPORTANT]
> Run this pipeline in a server. By renderization of the transfer function cause loss of the UI functionality.

## Build 
In ct-vr-docker:
```sh
docker-compose build
```

## Set path to data shared directory with docker image

The weights and legends are not provided within this repository. To generate new ones train with the `./train-val-network` folder. Then:

Set in /path/to/local/data:
- Models:
  * model/axis1/my_checkpoint/
	* model/axis2/my_checkpoint/
	* ...
- Legend:
	* model/legend.npy
- Dicom files in path (sample):
	* dicom/P064 (all dcm files inside this path)
- Transfer Functions in tf path:
	* tf/tf6.xml
- Copy ct-vr-docker/config.yaml to /path/to/local/data


## Execute:
Full pipeline
```sh
docker run -it --gpus all --network host --env="DISPLAY" --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,display" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/local/data/:/data/" ct-vr-docker_ctvr:latest python pipeline.py --full_pipeline  --dicom_path /data/dicom/P064
```

Full pipeline with video generator:
```sh
docker run -it --gpus all --network host --env="DISPLAY" --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,display" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/local/data/:/data/" ct-vr-docker_ctvr:latest python pipeline.py --full_pipeline  --dicom_path /data/dicom/P064 --video
```

## Citing

```bibtex
@software{ctvr-2025,
  title        = {A Deep Learning Framework for Pulmonary Disease Classification Using Volume-Rendered CTs},
  author       = {Noemi Maritza L. Romero, Ricco V. C. Soares, Mariana Recamonde-Mendoza, João L. D. Comba},
  year         = {2025},
  url          = {https://github.com/joaocomba/ct-vr}
}
```

## License
Apache License 2.0 (see `LICENSE`).
