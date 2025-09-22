import numpy as np
import pandas as pd
import glob
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import device
from tensorflow.keras.models import load_model 

import sys

class ModelCovidVR():
    def __init__(self, model_path, legend_path, slices_path, width, height):
        self.MODEL_PATH = model_path
        self.LEGEND_PATH = legend_path
        self.SLICES_PATH = slices_path
        self.WIDTH = width
        self.HEIGHT = height
        self.CLASSES = 2
    
    def test_patient(self, axis):
        try:
            result = {'success': True}
            prediction = []
            total_images = 0
            positive_count = []
            negative_count = []
            axis_detail = [] 
            axis_list = axis.split(",")
            
            class_legend = None
            if os.path.exists(self.LEGEND_PATH):
                class_legend = np.load(self.LEGEND_PATH, allow_pickle=True).item()
                class_legend = dict((v, k) for k,v in class_legend.items())
                self.CLASSES = len(class_legend.keys())

            count_by_class = dict([(key, []) for key in class_legend.keys()])
            
            for axis in axis_list:

                axis_name = f"axis{axis}"
                self.MODEL = load_model(os.path.join(self.MODEL_PATH, axis_name, "my_checkpoint"))
                patient_path = os.path.join(self.SLICES_PATH, axis_name)
                if not os.path.exists(patient_path):
                    return {'success': False, 'error': f'Path: {patient_path} does not exist'}

                imgs_filename = sorted(os.listdir(patient_path))
                test_filenames = imgs_filename[:] 
                test_df = pd.DataFrame({'filename': test_filenames})

                nb_samples = test_df.shape[0]
                total_images = nb_samples
                # DataGenerator:
                test_gen = ImageDataGenerator(rescale=1./255)
                test_generator = test_gen.flow_from_dataframe(
                    test_df,
                    patient_path,
                    x_col='filename',
                    y_col=None,
                    class_mode=None,
                    target_size=(self.WIDTH, self.HEIGHT),
                    batch_size=16,
                    shuffle=False
                )
                predict = self.MODEL.predict(test_generator, steps=np.ceil(nb_samples/16))
                
                if self.CLASSES == 2:
                    test_df['predicted'] = [ round(pr[0]) for pr in predict ]
                else:
                    test_df['predicted'] = [ np.where(pr == np.max(pr))[0][0] for pr in predict ]
                test_df['predicted'] = test_df['predicted'].replace(class_legend)
                test_df['count'] = 1
                test_df = test_df.groupby('predicted', as_index=False)['count'].count()
                
                axis_detail.append(test_df.iloc[0]["predicted"])
                for key in count_by_class.keys():
                    count_by_class[key].append(test_df.iloc[0]["count"])
                print(test_df)
            axis_length = len(axis_list)
            percentage_resume = dict([ ( key, sum(value) / (axis_length * total_images) ) for key, value in count_by_class.items()])
            result['resume'] = percentage_resume
            result['axis_detail'] = axis_detail
            result['axis_qty'] = axis_length
            result['predicted'] = max(percentage_resume.keys(), key=lambda x: percentage_resume[x])
            return result

        except Exception as e:
            et, eo, et = sys.exc_info()
            return {'success':False, 'error':f'error: {e}\n\n{et}\n\n{eo}\n\n{et}'}
            

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Test model.')

    parser.add_argument('--m', type=str, required=True, help='path to model')
    parser.add_argument('--l', type=str, required=True, help='path to legend')
    parser.add_argument('--s', type=str, required=True, help='path to slices')
    parser.add_argument('--a', type=str, required=True, help='Axis to use')
    parser.add_argument('--w', type=int, required=False, default=448, help='width')
    parser.add_argument('--h', type=int, required=False, default=448, help='height')

    args = parser.parse_args()
    model_covid_vr = ModelCovidVR(args.m, args.l, args.s, args.w, args.h)
    res = model_covid_vr.test_patient(args.a)
    with open('prediction_result.pkl', 'wb') as pr:
        pickle.dump(res, pr)
    del model_covid_vr

