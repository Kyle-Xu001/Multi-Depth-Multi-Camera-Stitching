import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from image_feature_extraction_test import Image
import image_match_test as match_utils

# Define the draw parameters for matching visualization
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)

# Manually define the ROI to locate the area for corresponding images
ROIs1 = [
    [300, 600, 600, 767],
    [150, 350, 300, 767]
]
ROIs2 = [
    [300, 150, 600, 420],
    [150, 0, 300, 420]
]


# load the matching images
img1 = cv.imread("lamp_19.JPG")
img2 = cv.imread("lamp_18.JPG")

Img1 = Image(img1)
Img2 = Image(img2)

kps1, des1 = Img1.findFeatures('sift')
kps2, des2 = Img2.findFeatures('sift')

masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)


kpsCluster1, desCluster1 = Img1.featureCluster(masks1)
kpsCluster2, desCluster2 = Img2.featureCluster(masks2)


matches = utils.clusterMatch(desCluster1,desCluster2)


img_match = cv.drawMatches(Img1.img,kpsCluster1[1],Img2.img,kpsCluster2[1],matches[1],None,**draw_params)


# Visualie the matching
plt.figure(1)
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

    


