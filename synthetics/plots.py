import matplotlib.pyplot as plt


def plot_synthetic(original, synthetic, figsize):
    """
    Plot original image, synthetic data, and difference.
    
    Parameters:
    - - - - -
    original: float, array
        original clean image slice
    synthetic: float, array
        synthetic, motion-corrupted image
    figsize: int, tuple
        size of figures
    """

    difference = original - synthetic

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=figsize)

    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Slice', fontsize=15)

    ax2.imshow(synthetic, cmap='gray')
    ax2.set_title('Synthetic Motion', fontsize=15)

    ax3.imshow(difference, cmap='gray')
    ax3.set_title('Difference', fontsize=15)

    return fig


def plot_motion_events(xmotion, ymotion, figsize):
    """
    Plot locations of motion in K-space.
    
    Parameters:
    - - - - -
    xmotion: float, array
        x-motion events
    ymotion: float, array
        y-motion events
    figsize: int, tuple
        size of figures
    """

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=figsize)

    ax1.spy(xmotion != 0, marker='o')
    ax1.set_title('X-Motion, K-Space', fontsize=15)

    ax2.spy(ymotion != 0, marker='o')
    ax2.set_title('Y-Motion, K-Space', fontsize=15)

    return fig