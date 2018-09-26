"""
Copyright(c) 2011, Cihat Eldeniz
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
                      SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
                      INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd

def makeFslXfmMatrix(T, R, S, filename):

    """
    Python conversion of original MATLAB implementation.
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
    tz = R[2]

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
        [0,0,0,1]])

    matrix = pd.DataFrame(M)
    matrix.to_csv(filename,index=False,header=None,sep=' ')