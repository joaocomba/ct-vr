## Build 
In covid-vr-docker:
```sh
docker-compose build
```

## Set path to data shared directory with docker image
Set in /path/to/local/data:
- Models:
    * model/axis1/my_checkpoint/
	* model/axis2/my_checkpoint/
	* ...
- Legend:
	*  model/legend.npy
- Dicom files in path (sample):
	* dicom/P064 (all dcm files inside this path)
- Transfer Functions in tf path:
	* tf/tf6.xml
- Copy covid-vr-docker/config.yaml to /path/to/local/data


## Execute:
Full pipeline
```sh
docker run -it --gpus all --network host --env="DISPLAY" --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,display" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/local/data/:/data/" covid-vr-docker_covidvr:latest python pipeline.py --full_pipeline  --dicom_path /data/dicom/P064
```

Full pipeline with video generator:
```sh
docker run -it --gpus all --network host --env="DISPLAY" --env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,display" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/local/data/:/data/" covid-vr-docker_covidvr:latest python pipeline.py --full_pipeline  --dicom_path /data/dicom/P064 --video
```

