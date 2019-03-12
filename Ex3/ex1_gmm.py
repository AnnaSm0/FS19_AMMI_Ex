from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import skimage

img = skimage.data.immunohistochemistry()[:80, -100:-20] / 255.

def gmm(data: np.ndarray, k: int) -> tuple:
    """
    Fit a Gaussian mixture model of k components to the given data.
    
    :param data: Nxd Numpy array (N d-dimensional data points)
    :param k: number of Gaussians
    :return: tuple (rs, means, covs, ws), where ``r`` is the Nxk array of
        responsibilities (gamma in the lecture slides), ``means`` is the kxd
        array of the components' mean values, ``covs`` is the kxdxd array of
        the components' covariance matrices, and ``ws`` is the k-element array
        of the components' weights.
    """
    raise NotImplementedError("Your implementation goes here.")


def plot_ellipses(means, covs, ax):
    """
    Plot the standard deviations of covariance matrices as ellipses.
    """
    # Following https://scikit-learn.org/0.15/auto_examples/mixture/plot_gmm_classifier.html
    for i in range(len(means)):
        
        v, w = np.linalg.eigh(covs[i])
        u = w[0] / np.linalg.norm(w[0])  # normalized 1st eigenvector
        angle = np.arctan2(u[1], u[0]) * 180 / np.pi  # convert to degrees
        s = np.sqrt(v) * 3  # 3 standard deviations
        ell = patches.Ellipse(means[i, :2], s[0], s[1], 180 + angle, facecolor="none", edgecolor="k")
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    

if __name__ == "__main__":
    
    k=4
    
    # Get the a and b channels from lab space
    pixels_ab = skimage.color.rgb2lab(img).reshape(-1, 3)[:, 1:]  # Nx2
    
    rs, means, covs, weights = gmm(data=pixels_ab, k=k)
    clusters = np.argmax(rs, axis=1)
    segmentation = clusters.reshape(img.shape[:2])
    
    plt.subplot(141)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    
    plt.subplot(142)
    plt.title("Pixel values in color space")
    plt.scatter(*pixels_ab.T, c=img.reshape(-1, 3))
    plt.xlabel("Channel a"); plt.ylabel("Channel b")
    
    ax = plt.subplot(143)
    plt.title("Found clusters (k={})".format(k))
    plt.scatter(*pixels_ab.T, c=clusters)
    plot_ellipses(means, covs, ax)
    plt.xlabel("Channel a"); plt.ylabel("Channel b")
    
    plt.subplot(144)
    plt.title("Segmentation")
    plt.imshow(segmentation)
    plt.show()
