"""
Plotting utilities for Bell state tomography.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_3d_hist_with_frame(data, ideal, title='', fidelity=None, purity=None):
    """
    Plot 3D histogram with frame for both real and imaginary parts.
    
    Parameters:
    -----------
    data : np.ndarray
        Density matrix data (4x4 complex array)
    ideal : np.ndarray
        Ideal density matrix (4x4 complex array)
    title : str
        Title for the figure
    fidelity : float, optional
        Fidelity value to display
    purity : float, optional
        Purity value to display
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': '3d'})
    # Create a grid of positions for the bars
    xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Create a custom colormap with two distinct colors for positive and negative values
    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # finding global min,max
    gmin = np.min([np.min(np.real(data)), np.min(np.imag(data)), np.min(np.real(ideal)), np.min(np.imag(ideal))])
    gmax = np.max([np.max(np.real(data)), np.max(np.imag(data)), np.max(np.real(ideal)), np.max(np.imag(ideal))])

    # Use the bar3d function with the 'color' parameter to color the bars
    for i in range(2):
        if i == 0:
            dz = np.real(data).ravel()
            dzi = np.real(ideal).ravel()
            axs[i].set_title('real')
        else:
            dz = np.imag(data).ravel()
            dzi = np.imag(ideal).ravel()
            axs[i].set_title('imaginary')
        axs[i].bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha=1, color=cmap(np.sign(dz)))
        axs[i].bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha=0.1, edgecolor='k')
        # Set tick labels for x and y axes
        axs[i].set_xticks(np.arange(1, 5))
        axs[i].set_yticks(np.arange(1, 5))
        axs[i].set_xticklabels(['00', '01', '10', '11'])
        axs[i].set_yticklabels(['00', '01', '10', '11'])
        axs[i].set_xticklabels(['00', '01', '10', '11'], rotation=45)
        axs[i].set_yticklabels(['00', '01', '10', '11'], rotation=45)
        axs[i].set_zlim([gmin, gmax])
        # Add fidelity and purity text to the first subplot (real part)
        if i == 0 and fidelity is not None and purity is not None:
            # Convert 3D axes coordinates to 2D for text placement
            axs[i].text2D(0.02, 0.98, f"Fidelity: {fidelity:.3f}\nPurity: {purity:.3f}",
                         transform=axs[i].transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.suptitle(title)
    # Show the plot

    return fig


def plot_3d_hist_with_frame_real(data, ideal, ax):
    """
    Plot 3D histogram with frame for real part on a given axis.
    
    Parameters:
    -----------
    data : np.ndarray
        Density matrix data (4x4 complex array)
    ideal : np.ndarray
        Ideal density matrix (4x4 complex array)
    ax : matplotlib.axes.Axes
        The 3D axes to plot on
    """
    xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Create a custom colormap with two distinct colors for positive and negative values
    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # finding global min,max
    gmin = np.min([np.min(np.real(data)), np.min(np.imag(data)), np.min(np.real(ideal)), np.min(np.imag(ideal))])
    gmax = np.max([np.max(np.real(data)), np.max(np.imag(data)), np.max(np.real(ideal)), np.max(np.imag(ideal))])

    # Use the bar3d function with the 'color' parameter to color the bars
    dz = np.real(data).ravel()
    dzi = np.real(ideal).ravel()
    ax.set_title('real')

    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha=1, color=cmap(np.sign(dz)))
    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha=0.1, edgecolor='k')
    # Set tick labels for x and y axes
    ax.set_xticks(np.arange(1, 5))
    ax.set_yticks(np.arange(1, 5))
    ax.set_xticklabels(['00', '01', '10', '11'])
    ax.set_yticklabels(['00', '01', '10', '11'])
    ax.set_xticklabels(['00', '01', '10', '11'], rotation=45)
    ax.set_yticklabels(['00', '01', '10', '11'], rotation=45)
    ax.set_zlim([gmin, gmax])
    # Show the plot


def plot_3d_hist_with_frame_imag(data, ideal, ax):
    """
    Plot 3D histogram with frame for imaginary part on a given axis.
    
    Parameters:
    -----------
    data : np.ndarray
        Density matrix data (4x4 complex array)
    ideal : np.ndarray
        Ideal density matrix (4x4 complex array)
    ax : matplotlib.axes.Axes
        The 3D axes to plot on
    """
    xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Create a custom colormap with two distinct colors for positive and negative values
    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # finding global min,max
    gmin = np.min([np.min(np.real(data)), np.min(np.imag(data)), np.min(np.real(ideal)), np.min(np.imag(ideal))])
    gmax = np.max([np.max(np.real(data)), np.max(np.imag(data)), np.max(np.real(ideal)), np.max(np.imag(ideal))])

    # Use the bar3d function with the 'color' parameter to color the bars
    dz = np.imag(data).ravel()
    dzi = np.imag(ideal).ravel()
    ax.set_title('imaginary')

    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha=1, color=cmap(np.sign(dz)))
    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha=0.1, edgecolor='k')
    # Set tick labels for x and y axes
    ax.set_xticks(np.arange(1, 5))
    ax.set_yticks(np.arange(1, 5))
    ax.set_xticklabels(['00', '01', '10', '11'])
    ax.set_yticklabels(['00', '01', '10', '11'])
    ax.set_xticklabels(['00', '01', '10', '11'], rotation=45)
    ax.set_yticklabels(['00', '01', '10', '11'], rotation=45)
    ax.set_zlim([gmin, gmax])
    # Show the plot

