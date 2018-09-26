import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as MVN


def kspace_trajectory(n, p):
    """
    Generate the K-space trajectory of visited voxels.
    
    Parameters:
    - - - - -
    n: int
        number of rows in input image
    p: int
        number of columns in input image
    """

    order = np.arange(n*p)

    for i in np.arange(1, p, 2):

        lo = i*p
        hi = (i+1)*p

        order[lo:hi] = np.flip(order[lo:hi], axis=0)

    return order


def mapping(coordinates, translation, size):
    """
    Translate a coordinate, based on observed motion pattern.
    
    Parameters:
    - - - - -
    coordinate: array of ints, N-coordinates x 2
        input coordinate to translate
    translation: array of ints, 1x2
        input translations for each dimension
    size: array of ints, 1x2
        Fourier domain size
        
    Return:
    - - - -
    [x, y]: array of ints
        mapped coordinates
    
    Examples:
    - - - - -
    coordinates = np.array([0, 0])
    translation = np.array([1, 1])
    size = np.array([4, 4])
    
    print(mapping(coordinates, translation, size))
    array([1,1])
    
    coordinates = np.array([1, 1])
    translation = np.array([2, 2])
    size = np.array([4, 4])
    
    print(mapping(coordinates, translation, size))
    array([3, 3])
    
    """

    # initialize translated coordinates array
    translated = np.zeros((coordinates.shape))
    # get number of coordinates, input image size, and translations
    [R, _] = coordinates.shape
    [sx, sy] = size
    [tx, ty] = translation

    # get original x and y coordinates
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # round movement to nearest integer
    # Notes:
    # bilinear interpolation method??
    x = np.round(x+tx)
    y = np.round(y+ty)

    # define boundaries of coordinate shifts
    if tx < 0:
        x = np.maximum(0, x)
    else:
        x = np.minimum(sx-1, x)

    if ty < 0:
        y = np.maximum(0, y)
    else:
        y = np.minimum(sy-1, y)

    translated[:, 0] = x
    translated[:, 1] = y

    return np.int32(translated)


def sample_motion(n_samples, indices, motion_df, img, xweight=None, yweight=None):
    """
    Sampel x and y motion for multivariate normal distribution.
    
    Parameters:
    - - - - -
    n_samples: int
        number of samples to generate
    indices: int, array
        indices to apply motion to
    motion_df: Pandas data frame
        data frame of sampled motion to fit 6d-Gaussian to
        expected column names: 'xrad', 'yrad', 'zrad'
                               'xshift', 'yshift', 'zshift'
    img: float, array
        input image
    xweight: float
        weight to apply to xmotion
    yweight: float
        weight to apply to ymotion
    """

    # get input image dimensions
    [n, p] = img.shape
    
    # get weights for each axis of motion
    if not xweight:
        xweight = 1
    if not yweight:
        yweight = 1

    # fit distribution from which to sample motion from
    mu = motion_df.mean(0)
    cov = motion_df.cov()
    gauss = MVN(mean=mu, cov= cov)

    # sample positions in K-space tracjectory for when
    # motion occurs in X and Y directions
    inds = np.random.choice(indices, size=n_samples, replace=False)
    inds = np.column_stack(np.unravel_index(inds, (n, p)))

    # sample motion for each coordinate
    samples = pd.DataFrame(
            gauss.rvs(size=n_samples),
            columns=motion_df.columns)

    x_motion = np.zeros((n, p))
    x_motion[inds[:, 0], inds[:, 1]] = xweight*np.asarray(samples['xshift'])
    x_coords = np.ravel_multi_index(np.where(x_motion != 0), x_motion.shape)

    y_motion = np.zeros((n, p))
    y_motion[inds[:, 0], inds[:, 1]] = yweight*np.asarray(samples['yshift'])
    y_coords = np.ravel_multi_index(np.where(y_motion != 0), y_motion.shape)

    return [x_motion, x_coords, y_motion, y_coords]


def fourier_shift(signal, order, shiftx, shifty, xcoords, ycoords):
    """
    Shift the K-space signal of an MRI slice.
    
    Parameters:
    - - - - - -
    signal: array, float
        input image
    order: list
        traversal order of pixel indices, analogous to the 
        K-space trajectory
    shiftx: array, float
        motion in x direction.  Same size as ```signal```.
        Each position experiences a potentially non-zero shift.
    shifty: array, float
        motion in y direction.  Same size as ```signal```.  
        Each position experiences a potentially non-zero shift.
    xcoords: list, int
        indices where motion occurs in x-direction
    ycoords: list, int
        indices where motion occurs in y-direction
        
    Returns:
    - - - - 
    motion: array, float
         spatial-domain signal with motion applied
    motion_fft: array, float
        frequency-domain signal with motion applied
        
    While the shifts in the x and y directions can theoretically be floats,
    here we make the assumption that motion occurs in discrete intervals.  
    If the degree of motion is a fraction of a pixel, we round to nearest int.

    """

    motion_coords = np.unique(np.concatenate([xcoords, ycoords]))

    ordr = list(order.copy())

    # Apply 2D-Fourier transform and shift k-space signal
    fft = np.fft.fft2(signal)
    fft_s = np.fft.fftshift(fft)

    [n, p] = fft_s.shape

    motion_fft = fft_s.copy()

    # Loop over indices in order
    for m in motion_coords:

        # convert indices to x-y coordinate pairs
        coords = np.column_stack(np.unravel_index(ordr[m:], (n, p)))
        [cx, cy] = coords[0, :]

        # get motion for current index
        trans_l = np.array([shiftx[cx, cy],
                            shifty[cx, cy]])

        # if motion in either x or y direction
        # apply shift to all coordinates
        if np.any(trans_l):

            # compute coordinate transforms
            mpc = mapping(coords, trans_l, np.array([n, p]))

            # update k-space information with new coordinates
            motion_fft[coords[:, 0], coords[:, 1]] = motion_fft[mpc[:, 0],
                                                                mpc[:, 1]]

    motion_ffts = np.fft.ifftshift(motion_fft)
    motion = np.abs(np.fft.ifft2(motion_ffts))

    return [motion, motion_fft]

def generate_outpath(filepath):

    """
    Generates the output name for an input nii.gz file by appending 'noise'
    to file name.
    """

    parts = filepath.split('.')
    prefix = '.'.join(parts[:-2]+['noise'])

    return prefix
