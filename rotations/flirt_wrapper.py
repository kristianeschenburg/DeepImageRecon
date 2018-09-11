import numpy as np
from nipype.interfaces import fsl

def flirt(in_image, ref_image, out_image, in_xfm):

    """
    Wrapper method to apply FSL FLIRT to images.

    Parameters:
    ----------
    in_image: string
        path to input image to transform
    ref_image: string
        path to reference image for transformation
    out_image: string
        path to output transformed image
    in_xfm: string
        path to transformation file
    """

    flt = fsl.FLIRT(in_file=in_image,
                    reference=ref_image,
                    out_file=out_image,
                    in_matrix_file=in_xfm,
                    apply_xfm=True)
    
    flt.run()
