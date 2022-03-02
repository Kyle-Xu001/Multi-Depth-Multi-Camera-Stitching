import cv2 as cv
import numpy as np

import utils
from image_feature_extraction_test import Image
import image_match_test as match_utils

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    
# load the matching images
img1 = cv.imread("lamp_16.JPG")
img2 = cv.imread("lamp_15.JPG")

Img1 = Image(img1)
Img2 = Image(img2)

Img1.equalizeHist()
Img2.equalizeHist()

kps1, des1 = Img1.findFeatures('sift')
kps2, des2 = Img2.findFeatures('sift')

ROIs1 = [
    [150, 350, 300, 767],
    [300, 600, 600, 767]
]
ROIs2 = [
    [150, 0, 300, 420],
    [300, 150, 600, 420]
]

masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)

kpsCluster1, desCluster1 = Img1.featureCluster(masks1)
kpsCluster2, desCluster2 = Img2.featureCluster(masks2)


