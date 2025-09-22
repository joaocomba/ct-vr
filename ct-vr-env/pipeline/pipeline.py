import pydicom
import glob
import os
import pandas as pd
import json
import requests
import pickle
import subprocess
import shutil
import time
import SimpleITK as sitk

from variables import *


def run_in_shell(command, is_python=False): 
    if is_python:
        process = subprocess.Popen(f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV_NAME} && echo $CONDA_DEFAULT_ENV && {command} && conda deactivate', shell=True, executable="/bin/bash")
    else:
        process = subprocess.Popen(f'export DISPLAY={GENERAL_DISPLAY} && {command}', shell=True,  executable="/bin/bash")
    process.wait()
    print(process.returncode)
    return True


def convert_dicom_to_nifti(dicom_path=None, nii_path=None, bin_path=None, normalize=True):
    # type: (str, str, bool) -> (bool, str)
    
    '''
    Developed by: Azael de Melo e Sousa
    04/07/2019
    Adapted by: Noemi Maritza Lapa Romero
    '''
    try:
        if not os.path.exists(dicom_path) or not os.path.isdir(dicom_path):
            return (False, 'Dicom dir  is NULL')

        nii_path = BASE_NII_ORIGINAL_OUTPUT_DIR if nii_path is None else nii_path
        nii_path = "{}.gz".format(nii_path) if nii_path.endswith(".nii") else nii_path
        if not nii_path.endswith(".nii.gz"):
            os.makedirs(nii_path, exist_ok=True)
        if os.path.isdir(nii_path):
            dicom_name = os.path.basename(os.path.normpath(dicom_path))
            nii_path = os.path.join(nii_path, "{}.nii.gz".format(dicom_name))
    	
        series_IDs = sitk.ImageSeriesReader_GetGDCMSeriesIDs(dicom_path)
        if not series_IDs:
            return (False, "No series in directory".format(dicom_path))
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_path, series_IDs[0])
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.LoadPrivateTagsOn()
        image3D=series_reader.Execute()

        sitk.WriteImage(image3D,nii_path)
        
        bin_path = BIN_PATH if bin_path is None else bin_path
        
        minimum = 1.0
        command = "%siftInterp %s VOXEL %f %f %f %s" % (bin_path, nii_path, minimum, minimum, minimum, nii_path)
        result = run_in_shell(command)
        if not result:
            return (False, "Error at iftInterp. Please compile the program iftInterp")
        
        return (True, nii_path)

    except Exception as e:
        return (False, f'Error: {e}')


def phnn_segmentation(nii_path=None, output_path=None, threshold=None, batch_size=None):
    try:
        if not os.path.exists(nii_path):
            return (False, f'File {nii_path} does not exist')
    
        study_id = nii_path.split("/")[-1]
        study_id = study_id.split(".")[0]

        output_path = BASE_SEGMENTED_OUTPUT_DIR if output_path is None else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            return (False, "Path {} does not exist".format(output_path))
    
        threshold = PHNN_THRESHOLD if threshold is None else threshold
        batch_size = PHNN_BATCH_SIZE if batch_size is None else batch_size
        
        command = f'{PYTHON_PATH} {PHNN_EXECUTABLE_PATH} --nii_path {nii_path} --directory_out {output_path} --batch_size {batch_size} --threshold {threshold}'
        print(command)
        phnn_seg = run_in_shell(command, is_python=True)
        print(phnn_seg)
        return (True, os.path.join(output_path, "crop_by_mask_{}.nii.gz".format(study_id)))
    except Exception as e:
        return (False, f'Error: {e}')


def mitk_views_maker(nii_path=None, output_path=None, tf_path=None, width=None, height=None, slices=None, axis=None, background_color=None):
    try:
        if nii_path is None:
            return (False, f'{nii_path} is NULL')
        if not os.path.exists(nii_path):
            return (False, f'Path {nii_path} does not exist')
        
        if not nii_path.endswith('.nii.gz'):
            return (False, f'{nii_path} is not .nii.gz file')
        
        study_id = nii_path.split('/')[-1]
        study_id = study_id[13:-7] # remove crop_by_mask_ and .nii.gz

        output_path = BASE_MITK_CAMERA_SHOT_OUTPUT_DIR if output_path is None else output_path
        output_path = os.path.join(output_path, study_id) if study_id not in output_path else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            return (False, f'Path {output_path} does not exist')
        tf_path = MITK_TRANSFER_FUNCTION_PATH if tf_path is None else tf_path
        if not os.path.exists(tf_path):
            return (False, f'Path {tf_path} does not exist')
    
        width = MITK_CAMERA_SHOT_WIDTH if width is None else width
        height = MITK_CAMERA_SHOT_HEIGHT if height is None else height
        slices = MITK_CAMERA_SHOT_LENGTH if slices is None else slices
        axis = MITK_CAMERA_SHOT_AXIS if axis is None else axis
        axis_str = ",".join([str(a) for a in axis])
        background_color = MITK_BACKGROUND_COLOR if background_color is None else background_color
        background_color_str = ",".join([str(c) for c in background_color])

        #### To get MITK_CAMERA_SHOT_LENGTH/2 will be odd and MITK_CAMERA_SHOT_LENGTH > 14, mandatory for script
        if slices % 2 == 1:
            slices +=1 
        if (slices / 2) % 2 == 0:
            slices += 2
        slices = max(14, slices)

        #### Iterate over nifti and build command
        command = f'vglrun {MITK_CAMERA_SHOT_EXECUTABLE_PATH} -tf {tf_path} -i {nii_path} -o {output_path} -w {width} -h {height} -s {slices} -a {axis_str} -c {background_color_str}'
        print(command)
        mitk_views = run_in_shell(command, is_python=False)
        print(mitk_views)
        return (True, output_path)
    except Exception as e:
        return (False, f'Error: {e}')


def mitk_video_maker(nii_path=None, output_path=None, tf_path=None, width=None, height=None, time=None, fps=None, background_color=None):
    try:
        if nii_path is None:
            return (False, f'{nii_path} is NULL')
        if not os.path.exists(nii_path):
            return (False, f'Path {nii_path} does not exist')
        if not nii_path.endswith('.nii.gz'):
            return (False, f'{nii_path} is not .nii.gz file')
        
        study_id = nii_path.split('/')[-1]
        study_id = study_id[13:-7] # remove crop_by_mask_ and .nii.gz

        output_path = BASE_MITK_CAMERA_SHOT_OUTPUT_DIR if output_path is None else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        if not os.path.exists(output_path):
            return (False, f'Path {output_path} does not exist')

        tf_path = MITK_TRANSFER_FUNCTION_PATH if tf_path is None else tf_path
        if not os.path.exists(tf_path):
            return (False, f'Path {tf_path} does not exist')
    
        width = MITK_VIDEO_WIDTH if width is None else width
        height = MITK_VIDEO_HEIGHT if height is None else height
        time = MITK_VIDEO_TIME if time is None else time
        fps = MITK_VIDEO_FPS if fps is None else fps
        background_color = MITK_BACKGROUND_COLOR if background_color is None else background_color
        background_color_str = ",".join([str(c) for c in background_color])
    
        video_path = os.join(output_path, "{}.mp4".format(study_id))
        #### Iterate over nifti and build command
        command = f'vglrun {MITK_VIDEO_EXECUTABLE_PATH} -tf {tf_path} -i {nii_path} -o {video_path} -w {width} -h {height} -t {time} -f {fps} -c {background_color_str}'
        print(command)
        mitk_views = run_in_shell(command, is_python=False)
        print(mitk_views)
        return (True, video_path)
    except Exception as e:
        return (False, f'Error: {e}')


def process_prediction(model_path=None, legend_path=None, slices_path=None, width=None, height=None, axis=None, classes=None):
    if slices_path is None:
        return (False, {'error': f'{slices_path} cannot be NULL'})
    
    model_path = PREDICTION_MODEL_PATH if model_path is None else model_path
    legend_path = PREDICTION_LEGEND_PATH if legend_path is None else legend_path

    print("model: {}, legend: {}".format(model_path, legend_path))
    if not os.path.exists(model_path):
        return (False, {'error': f'Path {model_path} does not exist'})
    if not os.path.exists(legend_path):
        return (False, {'error': f'Path {legend_path} does not exist'})
    
    width = PREDICTION_WIDTH if width is None else width
    height = PREDICTION_HEIGHT if height is None else height
    axis = AXIS if axis is None else axis
    axis_str = ",".join([str(a) for a in axis])
    
    try:
        #### Iterate over nifti and build command
        command = f'{PYTHON_PATH} models.py --m {model_path} --l {legend_path} --s {slices_path} --w {width} --h {height} --a {axis_str}'
        print(command)
        pred = run_in_shell(command, is_python=True)
        print(pred)
        result = None
        with open('prediction_result.pkl', 'rb') as pr:
            result = pickle.load(pr)

        if result is None or result['success'] is False:
            return (False, result)
        return (True, result)
    except Exception as e:
        return (False, {'error': f'Error: {e}'})


def run_convert_and_segment(dicom_path=None, nii_path=None, nii_segmented_path=None, bin_path=None):
    try:
        result = {'success': True, 'detail': '', 'nii_segmented_path': None}
        print('Init time: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL DICOM -> NIFTI
        study_id = None 
        bool_result, text_out = convert_dicom_to_nifti(dicom_path=dicom_path, nii_path=nii_path, bin_path=bin_path)
        if bool_result:
            nii_path = text_out
        else:
            result['success'] = False
            result['detail'] += f'Dicom to nifti: {text_out}'
            return result
        print('Convert dicom -> nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL PHNN SEGMENTATION
        bool_result, text_out = phnn_segmentation(nii_path=nii_path, output_path=nii_segmented_path, threshold=None, batch_size=None)
        if bool_result:
            nii_segmented_path = text_out
            result['nii_segmented_path'] = text_out
        else:
            result['success'] = False
            result['detail'] += f'Phnn segmentation: {text_out}'
        print('Segment nifti: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        return result

    except Exception as e:
        return {'success': False, 'detail': f'{e}'}


def run_ct_vr(nii_segmented_path=None, generate_video=False):
    try:
        result = {'success': True, 'detail': ''}
        print('Init time: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL MITK SLICE VIEWS
        views_path = None
        bool_result, text_out = mitk_views_maker(nii_path=nii_segmented_path, output_path=None, tf_path=None, width=None, height=None, slices=None, axis=None, background_color=None)
        if bool_result:
            views_path = text_out
        else:
            result['success'] = False
            result['detail'] += f'Slices 2d: {text_out}'
            return result
        print('slices 2d: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        if generate_video:
            # CALL MITK VIDEOS
            video_path = None
            bool_result, text_out = mitk_video_maker(nii_path=nii_segmented_path, output_path=None, tf_path=None, width=None, height=None, time=None, fps=None, background_color=None)
            if bool_result:
                video_path = text_out
                result['video_path'] = video_path
            else:
                result['success'] = False
                result['detail'] += f'Video: {text_out}'
                return result
            print('Videos: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # CALL PROCESS PREDICT
        bool_result, text_out = process_prediction(model_path=None, legend_path=None, slices_path=views_path, width=None, height=None, axis=None, classes=None)
        if bool_result:
            result['predicted'] = text_out['predicted']
            result['percentage'] = text_out['resume'][result['predicted']]
            result['axis_detail'] = text_out['axis_detail']
            result['axis_qty'] = text_out['axis_qty']
        else:
            result['success'] = False
            result['detail'] += f'\nPrediction process: {text_out}'
        print('Prediction: ' + time.strftime("%Y-%m-%d %H:%M:%S"))
        return result
    except Exception as e:
        return {'success': False, 'detail': f'{e}'}


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Run P-HNN on nii image.')

    parser.add_argument('--dicom_path', type=str, required=False, default=None, help='path to input dicom folder')
    parser.add_argument('--nii_out', type=str, required=False, default=None, help='path to output .nii file')
    parser.add_argument('--nii_segmented_path', type=str, required=False, default=None, help='Segmented nifti file path')
    parser.add_argument('--convert_and_segment', required=False, default=False, help='Process only convert and segmentation', action='store_true')
    parser.add_argument('--ct_vr_approach', required=False, default=False, help='Process only convert and segmentation', action='store_true')
    parser.add_argument('--full_pipeline', required=False, default=False, help='Process only prediction', action='store_true')
    parser.add_argument('--output', required=False, default='pipeline_result.json', help='file to save output, .json')
    parser.add_argument('--video', required=False, default=False, help='Generate video in process', action='store_true')
    
    args = parser.parse_args()

    result = None
    segmented_path = None
    
    if args.nii_segmented_path is not None:
        segmented_path = args.nii_segmented_path

    if args.full_pipeline is True or args.convert_and_segment is True:
        result = run_convert_and_segment(dicom_path=args.dicom_path, nii_path=args.nii_out, nii_segmented_path=args.nii_segmented_path)
        if result['success'] is True:
            segmented_path = result['nii_segmented_path']
        else:
            print(result)

    file_out = args.output
    with open(f'/data/{file_out}', 'w') as pr:
        json.dump(result, pr)
    
    if args.full_pipeline is True or args.ct_vr_approach is True:
        result = run_ct_vr(nii_segmented_path=segmented_path, generate_video=args.video)
        print(result)
    with open(f'/data/{file_out}', 'w') as pr:
        json.dump(result, pr)

