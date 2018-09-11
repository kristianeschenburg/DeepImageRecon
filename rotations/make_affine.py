import numpy as np

def makeFslXfmMatrix(T, R, S, filename):

    """
    Generate an FSL-compatible affine transformation matrix.

    Parameters:
    - - - - - 
    T: float, array
        translations in X, Y, and Z directions, in mm
    R: float, array
        rotations in X, Y, and Z axes, in radians
    S: float, array
        scale factor, unitless
    filename: string
        output name for .mat matrix
    """

    if not isinstance(T, np.ndarray):
        raise('Translation vector must be an array.')
    if not isinstance(R, np.ndarray):
        raise('Rotation vector must be an array.')
    if not isinstance(S, np.ndarray):
        raise('Scale factor vector must be an array.')
    if not isinstance(filename, str):
        raise('Filename must be a string.')
    
    tx = R[0]
    ty = R[1]
    tx = R[2]

    Rx = np.asarray([[1, 0, 0],
                     [0, np.cos(tx), np.sin(tx)],
                     [0, -np.sin(tx), np.cos(tx)]])
    Ry = np.asarray([[np.cos(ty), 0, -np.sin(ty)],
                     [0, 1, 0],
                     [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.asarray([[np.cos(tz), np.sin(tz), 0],
                     [-np.sin(tz), np.cos(tz), 0],
                     [0, 0, 1]])

    R3 = Rx.dot(Ry.dot(Rz))
    S3 = np.diag(S)

    RS = R3.dot(S3)
    M = np.row_stack(
        [np.column_stack([RS, T]), 
        [0,0,0,1])
