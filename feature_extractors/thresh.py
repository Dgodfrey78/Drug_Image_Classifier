import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data
from skimage.filters import try_all_threshold

img = imread('ima.jpg', as_gray=True)

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()
