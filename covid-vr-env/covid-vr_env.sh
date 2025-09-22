#!/bin/bash
conda init bash
source ~/.bashrc
conda activate covid-vr 
conda install pytorch==1.3.0 torchvision cudatoolkit==10.0 -c pytorch 
conda install -c caffe2 caffe
conda install dicom2nifti -c conda-forge
conda install pydicom
conda install nibabel
conda install requests
conda install flask
conda install -c simpleitk simpleitk
pip install tensorflow==2.4.0
pip install tensorflow-gpu==2.4.0


# conda install jupyter 
