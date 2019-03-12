import matplotlib.pyplot as plt
import numpy as np
import skimage

img = skimage.data.camera()

def otsu(img: np.ndarray) -> float:
    """
    Find the optimal threshold according to Otsu's method for the given image.
    
    :param img: two-dimensional Numpy array (grayscale image)
    :return: optimal threshold (maximum value of lower intensities)
    """
    raise NotImplementedError("Your implementation goes here.")
    

if __name__ == "__main__":
    
    threshold = otsu(img)
    
    plt.subplot(121)
    plt.title("Original image")
    plt.imshow(img, cmap="gray")
    plt.subplot(122)
    plt.title("Segmentation with threshold {}".format(threshold))
    plt.imshow(img > threshold, cmap="gray")
