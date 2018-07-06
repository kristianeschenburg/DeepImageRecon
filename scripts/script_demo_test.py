import argparse, datetime, os, yaml
import sys
sys.path.append('../cafndl/')

from glob import glob

import nibabel as nb
import numpy as np
from scipy import io as sio

from cafndl_fileio import *
from cafndl_utils import *
from cafndl_network import *
from cafndl_metrics import *

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import model_from_json


parser = argparse.ArgumentParser()
parser.add_argument('-config','--configuration',
                    help='Configuration file with parameters.',
                    required=True,type=str)
args = parser.parse_args()


""" Load configuration file/ """
with open(args.configuration,'r') as ymlfile:
    cfg = yaml.load(ymlfile)

for attribute,params in cfg.items():
    print(attribute)
    print(params)

    
""" Load input parameters """
subjects = glob(''.join([cfg['test_data_load']['input_directory'],'*/']))

noise_file = cfg['test_data_load']['noise']
truth_file = cfg['test_data_load']['truth']

testing_files = [{'noise': ''.join([subj,noise_file]),
					'truth': ''.join([subj,truth_file])} for subj in subjects]

num_dataset_test = len(testing_files)                
print('process {0} data description'.format(num_dataset_test))


""" Loop over each noisy / clean image pair """
list_test_noise = []
list_test_truth = []
list_augments = []
for index_data in range(num_dataset_test):

	# Get noisy image file path
    ptrn_noise = testing_files[index_data]['noise']
    print('Noise image: {:}'.format(ptrn_noise))
    img_noise = nb.load(ptrn_noise).get_data()
    [nx,ny,nz] = img_noise.shape

    # Get clean image file path
    ptrn_truth = testing_files[index_data]['truth']
    print('Ground truth image: {:}'.format(ptrn_truth))
    img_truth = nb.load(ptrn_truth).get_data()
    [tx,ty,tz] = img_truth.shape

    list_test_noise.append(prepare_data_from_nifti(img_noise, list_augments))
    list_test_truth.append(prepare_data_from_nifti(img_truth, list_augments))

# generate test dataset
scale_data = 100.
data_test_noise = scale_data * np.concatenate(list_test_noise, axis = 0)
data_test_truth = scale_data * np.concatenate(list_test_truth, axis = 0)    
data_test_residual = data_test_truth - data_test_noise

print('mean, min, max')
print(np.mean(data_test_noise.flatten()),np.min(data_test_noise.flatten()),np.max(data_test_noise.flatten()))
print(np.mean(data_test_truth.flatten()),np.min(data_test_truth.flatten()),np.max(data_test_truth.flatten()))
print(np.mean(data_test_residual.flatten()),np.min(data_test_residual.flatten()),np.max(data_test_residual.flatten()))
print('generate test dataset with augmentation size {0},{1}'.format(data_test_noise.shape, data_test_truth.shape))



""" Load model parameters """
modelparams = cfg['model']
modelfile = ''.join([modelparams['modeldir'],'model_demo.',modelparams['modelname'],'.json'])
modelweights = ''.join([modelparams['modeldir'],'model_demo.',modelparams['modelname'],'.weights.json'])

json_file = open(modelfile,'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(modelweights)

# Number of input and output channels
num_channel_input = data_test_noise.shape[-1]
num_channel_output = data_test_truth.shape[-1]

# Expected input dimensionality
img_rows = data_test_noise.shape[1]
img_cols = data_test_truth.shape[1]


batch_size=1

'''
apply model
'''

t_start_pred = datetime.datetime.now()
data_test_output = model.predict(data_test_noise, batch_size=batch_size)

# clamp
clamp_min = -0.5
clamp_max = 0.5
data_test_output = np.maximum(np.minimum(data_test_output, clamp_max), clamp_min)

# add
data_test_output += data_test_noise

t_end_pred = datetime.datetime.now()
print('predict on data size {0} using time {1}'.format(
	data_test_output.shape, t_end_pred - t_start_pred))

### Load output parameters:
outputparams = cfg['output']
outputdir = outputparams['prediction_dir']

if not os.path.exists(outputdir):
	os.mkdir(outputdir)

filename_results = ''.join([outputdir,outputparams['filename_results']])

'''
export images
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
num_sample = data_test_output.shape[0]
list_err_pred = []
list_err_noise = []
aug_err = 10
for i in range(num_sample):
    
    # get image
    im_truth = np.squeeze(data_test_truth[i,:,:,0]).T
    im_noise = np.squeeze(data_test_noise[i,:,:,0]).T
    im_pred = np.squeeze(data_test_output[i,:,:,0]).T

	# get error
    err_pred = getErrorMetrics(im_pred, im_truth)
    err_noise = getErrorMetrics(im_noise, im_truth)
    list_err_pred.append(err_pred)
    list_err_noise.append(err_noise)
	
    # display
    im_toshow = [im_truth, im_noise, (im_noise-im_truth)*aug_err, im_pred, (im_pred-im_truth)*aug_err]
    im_toshow = np.abs(np.concatenate(im_toshow, axis=1))
    plt.figure(figsize=[20,8])
    plt.imshow(im_toshow, clim=[0,0.5], cmap='gray')
    im_title = 'sample #{0}, input PSNR {1:.4f}, SSIM {2:.4f}, predict PSNR {3:.4f}, SSIM {4:.4f}'.format(
				i, err_noise['psnr'], err_noise['ssim'], err_pred['psnr'], err_pred['ssim'])
    plt.title(im_title)
    print(im_title)
    path_figure = filename_results+'_{0}.png'.format(i)
    plt.savefig(path_figure)
    plt.close()

'''
export results
'''
print('input average metrics:', {k:np.mean([x[k] for x in list_err_noise]) for k in list_err_noise[0].keys()})
print('prediction average metrics:', {k:np.mean([x[k] for x in list_err_pred]) for k in list_err_pred[0].keys()})# save history dictionary
import json
result_error = {'err_noise':list_err_noise,
				'err_pred':list_err_pred}
path_error = filename_results+'_error.json'
with open(path_error, 'w') as outfile:
    json.dump(result_error, outfile)
print('error exported to {0}'.format(path_error))




