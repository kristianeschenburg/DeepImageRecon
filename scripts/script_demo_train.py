from scipy import io as sio
import numpy as np
import os
import dicom
import nibabel as nib
import datetime

import sys
sys.path.append('../cafndl/')
from cafndl_fileio import *
from cafndl_utils import *
from cafndl_network import *
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',help='Training epochs.',
	required=False,type=int,default=100)

parser.add_argument('--modelname',help='Output file name to save model, coefficients, and epoch images.',
	required=False,type=str,default=None)

parser.add_argument('--byslice',help='Treat each slice as a training image.',
	required=False,type=bool,choices=[True,False],default=False)

parser.add_argument('--loss',help='Loss function to use.',required=False,
	type=str,default='mean_absolute_error',choices=['mean_squared_error',
	'mean_absolute_error','kullback_leibler_divergence',
	'mean_absolute_percentage_error'])

args = parser.parse_args()

num_epoch = args.epochs

# input noisy / clean image pairs
# constitute the entire set of training data
list_training_data = [{
        'inputs': '../test_data/subj_1/AX_Flair_Clear.nii.gz',
        'gt': '../test_data/subj_1/T2_Flair_Sense.nii.gz'},
        {'inputs': '../test_data/subj_2/AX_Flair_Clear.nii.gz',
        'gt': '../test_data/subj_2/T2_Flair_Sense.nii.gz'
        }]

num_dataset_train = len(list_training_data)                
print('process {0} data description'.format(num_dataset_train))


# Build augmentation parameter list
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
list_train_input = []
list_train_gt = []

# Loop over each noisy / clean image pair
for index_data in range(num_dataset_train):

	# get noisy image file path
    path_train_noise = list_training_data[index_data]['inputs']
    print('Noise image: {:}'.format(path_train_noise))
    path_image_noise = nb.load(path_train_noise)
    train_noise = path_image_noise.get_data()
    [nx,ny,nz] = train_noise.shape

    # get clean image file path
    path_train_truth = list_training_data[index_data]['gt']
    print('Ground truth image: {:}'.format(path_train_truth))
    path_image_truth = nb.load(path_train_noise)
    train_truth = path_image_noise.get_data()
    [tx,ty,tz] = train_truth.shape
    
    # if using each slice to train
    if args.byslice:

    	for zslice in np.arange(nz):
    		train_noise_slice = prepare_data_from_nifti(train_noise, list_augments,slices=[zslice])
    		list_train_input.append(train_noise_slice)

    	for zslice in np.arange(tz):
    		train_truth_slice = prepare_data_from_nifti(train_truth, list_augments,slices=[zslice])
    		list_train_gt.append(train_truth_slice)

    # otherwise load whole augmented volumes
    else:
	    # load data
	    data_train_input = prepare_data_from_nifti(train_noise, list_augments)
	    data_train_gt = prepare_data_from_nifti(train_truth, list_augments)
    
	    # append
	    list_train_input.append(data_train_input)
	    list_train_gt.append(data_train_gt)


# generate and scale dataset    
scale_data = 100.
data_train_input = scale_data * np.concatenate(list_train_input, axis = 0)
data_train_gt = scale_data * np.concatenate(list_train_gt, axis = 0)    
data_train_residual = data_train_gt - data_train_input

# data_train_input / data_train_gt are both of size (n-slices * n-augmentations) x X x Y x (n-channels -- should be 1)

print('mean, min, max')
print(np.mean(data_train_input.flatten()),np.min(data_train_input.flatten()),np.max(data_train_input.flatten()))
print(np.mean(data_train_gt.flatten()),np.min(data_train_gt.flatten()),np.max(data_train_gt.flatten()))
print(np.mean(data_train_residual.flatten()),np.min(data_train_residual.flatten()),np.max(data_train_residual.flatten()))
print('generate train dataset with augmentation size {0},{1}'.format(
	data_train_input.shape, data_train_gt.shape))


'''
setup parameters
'''
# related to model
num_poolings = 3
num_conv_per_pooling = 3

# learning rate, related to training
lr_init = 0.001

#num_epoch = 100
ratio_validation = 0.1
batch_size = 4

# default settings
num_channel_input = data_train_input.shape[-1]
num_channel_output = data_train_gt.shape[-1]
img_rows = data_train_input.shape[1]
img_cols = data_train_gt.shape[1]
keras_memory = 0.4
keras_backend = 'tf'
with_batch_norm = True
print('setup parameters')


'''
init model
'''

if args.modelname:
	outname = ''.join(['.',args.modelname,'.'])
else:
	outname = ''

filename_checkpoint = '../checkpoints/model_demo_0001'
filename_checkpoint = ''.join([filename_checkpoint,outname,'.',str(num_epoch),'.ckpt'])

filename_init = '../checkpoints/model_demo'
filename_init = ''.join([filename_init,outname,'.',str(num_epoch),'.ckpt'])

filename_model = '../checkpoints/model_demo'
filename_model = ''.join([filename_model,outname,'.',str(num_epoch),'.json'])

filename_modelweights = '../checkpoints/model_demo'
filename_modelweights = ''.join([filename_modelweights,outname,'.weights.',str(num_epoch),'.h5'])


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
						with_bn = with_batch_norm, verbose=1,
						loss_function=args.lossfunctin)
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
history = model.fit(data_train_input, data_train_residual,
			batch_size = batch_size,
			epochs = num_epoch,
			verbose = 1,
			shuffle = True,
			callbacks = [callback_checkpoint],
			validation_split = ratio_validation)
t_end_train = datetime.datetime.now()
print('finish training on data size {0} for {1} epochs using time {2}'.format(
		data_train_input.shape, num_epoch, t_end_train - t_start_train))

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
