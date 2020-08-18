## Bulk image resizer

# This script simply resizes all the images in a folder to one-eigth their
# original size. It's useful for shrinking large cell phone pictures down
# to a size that's more manageable for model training.

# Usage: place this script in a folder of images you want to shrink,
# and then run it.

import numpy as np
import cv2
import os

dir_path = "C:/tensorflow1/models/research/object_detection/images/test/"

for filename in os.listdir(dir_path):
    # If the images are not .JPG images, change the line below to match the image type.
    if filename.endswith(".png"):
        print(filename)
        image = cv2.imread(filename)
        resized = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        cv2.imwrite(filename,resized)
