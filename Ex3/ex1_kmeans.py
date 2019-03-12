import matplotlib.pyplot as plt
import numpy as np
import skimage

img = skimage.data.immunohistochemistry()[:80, -100:-20] / 255.

def kmeans(data: np.ndarray, k: int, random_seed: int) -> np.ndarray:
    """
    Cluster the given data into k clusters using the k-means algorithm.
    
    :param data: Nxd Numpy array (N d-dimensional data points)
    :param k: as the name implies
    :param random_seed: for deterministic random choice of initial cluster centers
    :return: N-element Numpy array, each value in {0,...,k-1}, denoting the
        respective sample's cluster membership.
    """
    # Initialize the cluster centers by randomly drawing k points from the data
    r = np.random.RandomState(seed=random_seed)
    centers = data[r.choice(len(data), size=k, replace=False)]  # kxd
    
    raise NotImplementedError("Your remaining implementation goes here.")
    

if __name__ == "__main__":
    
    k=4
    
    # Get the a and b channels from lab space
    pixels_ab = skimage.color.rgb2lab(img).reshape(-1, 3)[:, 1:]  # Nx2
    
    clusters = kmeans(data=pixels_ab, k=k, random_seed=42)
    segmentation = clusters.reshape(img.shape[:2])
    
    plt.subplot(141)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    
    plt.subplot(142)
    plt.title("Pixel values in color space")
    plt.scatter(*pixels_ab.T, c=img.reshape(-1, 3))
    plt.xlabel("Channel a"); plt.ylabel("Channel b")
    
    plt.subplot(143)
    plt.title("Found clusters (k={})".format(k))
    plt.scatter(*pixels_ab.T, c=clusters)
    plt.xlabel("Channel a"); plt.ylabel("Channel b")
    
    plt.subplot(144)
    plt.title("Segmentation")
    plt.imshow(segmentation)
    plt.show()
