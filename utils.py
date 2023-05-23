import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import torch


def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def show_image_w_label(image, label,cmap='gray'):
    #image[0] is to get the values from the tensor
    if type(image) == torch.Tensor:
        #permute tensor to get channels last
        image = image.permute(1,2,0)
        plt.imshow(image, cmap=cmap)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(label)
    plt.show()
