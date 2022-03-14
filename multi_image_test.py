import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from image_feature_extraction_test import Image
import image_match_test as match_utils

# Define the draw parameters for matching visualization
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(0, 0, 255),
                   flags=cv.DrawMatchesFlags_DEFAULT)

# load the matching images
img1 = cv.imread("lamp_18.JPG")
img2 = cv.imread("lamp_17.JPG")
img3 = cv.imread("lamp_16.JPG")
img4 = cv.imread("lamp_15.JPG")
img5 = cv.imread("lamp_14.JPG")

homo_mat4 = np.array([
    [1.185309268822935, 0.02780698568974888, -126.38706443249286],
    [0.08754572139261853, 1.173773247849686, 410.4095462935114],
    [0.0001292217749010845, 0.00008411840350431905, 1.0]])

homo_mat1 = np.array([
    [1.3101417857973252,0.18817547106671775,-82.05548599904998],
    [-0.08741143216119451,1.4365538609794652,562.4395848794102],
    [0.0003070468993624202,0.00029575251347828337,0.9999999999999999]])

# Get the position of vertices
posVerts = utils.transformVerts(img_size=np.array(
    [img4.shape[1], img4.shape[0]]), homo_mat=homo_mat1)
# print("Left Top: ",posVerts[0,:],"\n",
#       "Right Top: ",posVerts[1,:],"\n",
#       "Right Bottom: ",posVerts[2,:],"\n",
#       "Left Bottom: ",posVerts[3,:],"\n")

x_min = posVerts[:, 0].min()
x_max = posVerts[:, 0].max()
y_min = posVerts[:, 1].min()
y_max = posVerts[:, 1].max()
print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %
      (x_min, x_max, y_min, y_max))

stitch_size = (x_max, y_max)

homo_mat_ = np.eye(3)
img_super = cv.warpPerspective(
    img5, homo_mat_, stitch_size, borderValue=(0, 0, 0))
img_transform = cv.warpPerspective(
    img4, homo_mat1, stitch_size, borderValue=(0, 0, 0))

# Combine the image on one super image
high_y = np.min(posVerts[:, 1])
img_transform[high_y:high_y, :, :] = 0
img_super[img_transform > 0] = 0

img_stitch = img_transform + img_super

plt.figure(3)
plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
