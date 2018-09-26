import argparse
import nibabel as nb
import numpy as np
import pandas as pd

import motion_utilities as mutl
import plots
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input volume to add synthetic noise to.',
    required=True, type=str)
parser.add_argument('-n', '--noise', help='Motion parameters file.',
    required=True, type=str)
parser.add_argument('-xw', '--xweight', help='Weight to apply to motion in X.',
    required=False, default=25, type=float)
parser.add_argument('-yw', '--yweight', help='Weight to apply to motion in Y.',
    required=False, default=25, type=float)
parser.add_argument('-m', '--motion', help='Number of motion events.',
    required=False, default=25, type=int)
parser.add_argument('-v', '--verbose', help='Output figures of image slices.',
    required=False, default=True, type=bool)

args = parser.parse_args()

# get parameters
input_image = args.input
noise = args.noise
x_weight = args.xweight
y_weight = args.yweight
events = args.motion
verbose = args.verbose

# load volume
image = nb.load(input_image)
data = image.get_data()
print('Input image shape: {:}'.format(data.shape))
print('Orienting image in A-P direction.')
data = np.fliplr(data.T)

# initialize synthetic array
[x, y, z] = data.shape
synthetic = np.zeros((x, y, z))

# compute K-space trajectory
order = mutl.kspace_trajectory(y, z)

# load motion parameters from which to sample motion from
print('Loading sample motion.')
motion_params = pd.read_table(
    noise, names=['xrad', 'yrad', 'zrad',
                  'xshift', 'yshift', 'zshift'],
    sep='\s+')

# generate output name prefix
prefix = mutl.generate_outpath(input_image)

# loop over image slices, adding noise to each
print('Looping over image slices.')
for img_slice in np.arange(x):

    # get current slice
    img = data[img_slice, :, :]

    # generate unique motion for slice
    [xm, xc, ym, yc] = mutl.sample_motion(
        events, order, motion_params, img, 
        xweight=x_weight, yweight=y_weight)

    # generate synthetically noisy image
    [motion, motion_fft] = mutl.fourier_shift(
            signal=img, order=order,
            shiftx=xm, shifty=ym,
            xcoords=xc, ycoords=yc)

    # fill motion array
    synthetic[img_slice, :, :] = motion

    # if verbose, generate figues of ground truth, motion, and difference
    if verbose and img_slice % 20 == 0:
        slice_prefix = ''.join([prefix,'.slice.{:}.jpg'.format(img_slice)])
        G = plots.plot_synthetic(img, motion, (12, 8))
        plt.savefig(slice_prefix)
        plt.close()


# save synthetic image
synthetic = np.fliplr(synthetic).T
nii_image = nb.nifti1.Nifti1Image(
    synthetic, affine=image.affine, header=image.header)

output_file = '.'.join([prefix,'nii.gz'])
nb.save(img=nii_image, filename=output_file)
