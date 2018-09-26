import numpy as np
import nibabel as nb

def add(original, in_images, weights, filename=None):

    """
    Method to combine linearly combine set of images.

    Parameters:
    - - - - -
    original: string
        input image file to add corruption to
    in_images: list, string
        images to corrupt original image with
    weights: array, float
        weights to apply to each corrupting image
    filename: string
        output filename of fully corrupted image

    Returns:
    - - - -
    corrupted: array
        noisy, corrupted original image volume
    """

    og = nb.load(original)
    og_data = og.get_data()

    corrupted = np.zeros((og_data.shape)) + og_data

    for ig, weight in zip(in_images, weights):

        ig_obj = nb.load(ig)
        ig_data = ig_obj.get_data()
        corrupted += (ig_data*weight)

    return [og_data, corrupted]