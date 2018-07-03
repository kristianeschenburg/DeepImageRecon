import dicom
import nibabel as nb
import numpy as np
from cafndl_utils import augment_data

def prepare_data_from_nifti(data_volume, list_augments=[], scale_by_norm=True, slices=None):

	"""
    Parameters:
    - - - - -
        data_volume: input volume
        list_augments: types of volumetric transformations to apply (flipx/y, flipxy, shiftx/y)
        scale_by_norm: normalize the volume data
        slices: specific slices to augment. |slices| = number of slices in original volume Z-dimension.
    """

	# transpose to slice*x*y*channel (slice = z-dimension)
	if np.ndim(data_volume)==3:
		data_volume = data_volume[:,:,:,np.newaxis]
	data_volume = np.transpose(data_volume, [2,0,1,3])
	
	# scale
	if scale_by_norm:
		data_volume = data_volume / np.linalg.norm(data_volume.flatten())
	
	# extract slices
	if slices is not None:
		data_volume = data_volume[slices,:,:,:]

	# finish loading data
	print('Image loaded, data size {:} (sample, x, y, channel)'.format(data_volume.shape))    

	
	# augmentation
	if len(list_augments)>0:

		list_data = []

		for augment in list_augments:
			data_augmented = augment_data(data_volume, axis_xy = [1,2], augment = augment)
			list_data.append(data_augmented.reshape(data_volume.shape))

		data_volume = np.concatenate(list_data, axis = 0)

	return data_volume