import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def show_image_w_label(image, label,cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(label)
    plt.show()
