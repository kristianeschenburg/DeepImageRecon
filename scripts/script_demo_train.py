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
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam




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
subjects = glob(''.join([cfg['data_load']['input_directory'],'*/']))

noise_file = cfg['data_load']['noise']
truth_file = cfg['data_load']['truth']

training_files = [{'noise': ''.join([subj,noise_file]),
					'truth': ''.join([subj,truth_file])} for subj in subjects]

by_slice = cfg['data_load']['by_slice']

num_dataset_train = len(training_files)                
print('process {0} data description'.format(num_dataset_train))
    

""" Load output parameters """
outname = cfg['data_save']['modelname']
outdirc = cfg['data_save']['outputdir']


""" Build augmentation parameter list, don't really change this """
list_augments = []
num_augment_flipxy = 2
num_augment_flipx = 2
num_augment_flipy = 2
num_augment_shiftx = 1
num_augment_shifty = 1
for flipxy in range(num_augment_flipxy):
	for flipx in range(num_augment_flipx):
		for flipy in range(num_augment_flipy):
			for shiftx in range(num_augment_shiftx):
				for shifty in range(num_augment_shifty):
					augment={'flipxy':flipxy,'flipx':flipx,'flipy':flipy,'shiftx':shiftx,'shifty':shifty}
					list_augments.append(augment)
num_augment=len(list_augments)
print('will augment data with {0} augmentations'.format(num_augment))


# Generate training data by applying augmentation transformations to each image slice
list_train_noise = []
list_train_truth = []


""" Loop over each noisy / clean image pair """
for index_data in range(num_dataset_train):

	# Get noisy image file path
    ptrn_noise = training_files[index_data]['noise']
    print('Noise image: {:}'.format(ptrn_noise))
    img_noise = nb.load(ptrn_noise).get_data()
    [nx,ny,nz] = img_noise.shape

    # Get clean image file path
    ptrn_truth = training_files[index_data]['truth']
    print('Ground truth image: {:}'.format(ptrn_truth))
    img_truth = nb.load(ptrn_truth).get_data()
    [tx,ty,tz] = img_truth.shape
    
    # If training slice by slice
    if by_slice:
        for zslice in np.arange(nz):
            train_noise_slice = prepare_data_from_nifti(img_noise, list_augments,slices=[zslice])
            list_train_noise.append(train_noise_slice)
    
        for zslice in np.arange(tz):
            train_truth_slice = prepare_data_from_nifti(img_truth, list_augments,slices=[zslice])
            list_train_truth.append(train_truth_slice)

    # Otherwise load whole augmented volumes
    else:
	    list_train_noise.append(prepare_data_from_nifti(img_noise, list_augments))
	    list_train_truth.append(prepare_data_from_nifti(img_truth, list_augments))


""" Generate and scale dataset """
scale_data = 100.
data_train_noise = scale_data * np.concatenate(list_train_noise, axis = 0)
data_train_truth = scale_data * np.concatenate(list_train_truth, axis = 0)    
data_train_resid = data_train_truth - data_train_noise


""" 
Data_train_noise / data_train_truth
both of size (n-slices * n-augmentations) x X x Y x (n-channels -- should be 1)
"""
print('mean, min, max')
print(np.mean(data_train_noise.flatten()),np.min(data_train_noise.flatten()),
      np.max(data_train_noise.flatten()))
print(np.mean(data_train_truth.flatten()),np.min(data_train_truth.flatten()),
      np.max(data_train_truth.flatten()))
print(np.mean(data_train_resid.flatten()),np.min(data_train_resid.flatten()),
      np.max(data_train_resid.flatten()))

print('generate train dataset with augmentation size {0},{1}'.format(
	data_train_noise.shape, data_train_truth.shape))


""" Load model parameters """
modelparams = cfg['model']
modelfile = ''.join([modelparams['modeldir'],'model_demo.',modelparams['modelname'],'.json'])
modelweights = ''.join([modelparams['modeldir'],'model_demo.',modelparams['modelname'],'.weights.json'])

# Number of input and output channels
num_channel_input = data_test_noise.shape[-1]
num_channel_output = data_test_truth.shape[-1]

# Expected input dimensionality
img_rows = data_test_noise.shape[1]
img_cols = data_test_truth.shape[1]


# Default settings related to Keras, don't change
keras_memory = 0.4
keras_backend = 'tf'
with_batch_norm = True
print('setup parameters')


'''
init model
'''

filename_checkpoint = ''.join([outdirc,'model_demo.',outname,'.ckpt'])
filename_model = ''.join([outdirc,'model_demo.',outname,'.json'])
filename_modelweights = ''.join([outdirc,'model_demo',outname,'.weights.json'])

filename_init = ''

callback_checkpoint = ModelCheckpoint(filename_checkpoint, 
								monitor='val_loss', 
								save_best_only=True)
setKerasMemory(keras_memory)
model = deepEncoderDecoder(num_channel_input = num_channel_input,
                           num_channel_output = num_channel_output,
                           img_rows = img_rows,
                           img_cols = img_cols,
                           lr_init = lr_init,
                           num_poolings = num_poolings,
                           num_conv_per_pooling = num_conv_per_pooling,
                           batch_norm = batch_norm,
                           verbose=1,
                           loss_function=loss_function)

print('train model:', filename_checkpoint)
print('parameter count:', model.count_params())


'''
train network
'''
try:
	model.load_weights(filename_init)
	print('model trains from loading ' + filename_init)        
except:
	print('model trains from scratch')

model.optimizer = Adam(lr = lr_init)
t_start_train = datetime.datetime.now()

history = model.fit(data_train_noise,
                    data_train_resid,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    shuffle = True,
                    callbacks = [callback_checkpoint],
                    validation_split = validation_split)

t_end_train = datetime.datetime.now()
print('finish training on data size {0} for {1} epochs using time {2}'.format(
		data_train_noise.shape, epochs, t_end_train - t_start_train))

# serialize model to JSON
model_json = model.to_json()
with open(filename_model,'w') as json_file:
    json_file.write(model_json)
model.save_weights(filename_modelweights)
print("Saved model to disk.")

'''
save training results
'''
# save train loss/val loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
path_figure = filename_checkpoint+'.png'
plt.savefig(path_figure)

# save history dictionary
import json
path_history = filename_checkpoint+'.json'
with open(path_history, 'w') as outfile:
    json.dump(history.history, outfile)
