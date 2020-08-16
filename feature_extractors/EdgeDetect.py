
#importing the required libraries
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image


import pprint
pp = pprint.PrettyPrinter(indent=4)
from skimage.feature import hog

from skimage.transform import rescale
import scipy.misc
import os



def edge(img):
    #reading the image
    data = imread('ima.jpg',as_gray=True)
    #datas=img
    data=rgb2gray(data)


    #calculating horizontal edges using prewitt kernel
    edges_prewitt_horizontal = prewitt_h(data)
    #calculating vertical edges using prewitt kernel
    edges_prewitt_vertical = prewitt_v(data)

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8,6)
    # remove ticks and their labels
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
   
 

    ax.imshow(edges_prewitt_vertical, cmap='gray')
    ax.set_title('hog')

    fig.savefig('fuck2w.jpg',bbox_inches='tight')

    plt.show()



    image = cmap(norm(data))
    

    plt.imshow(image, cmap='gray')
    return edges_prewitt_vertical


