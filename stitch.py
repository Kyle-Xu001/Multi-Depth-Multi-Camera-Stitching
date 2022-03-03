import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from image_feature_extraction_test import Image
import image_match_test as match_utils

# Define the draw parameters for matching visualization
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (0,0,255),
                   flags = cv.DrawMatchesFlags_DEFAULT)

# Manually define the ROI to locate the area for corresponding images
ROIs1 = [
    [150, 350, 300, 767],
    [300, 500, 1000, 767]]

ROIs2 = [
    [150, 0, 300, 420],
    [300, 0, 1000, 420]]


# load the matching images
img1 = cv.imread("lamp_16.JPG")
img2 = cv.imread("lamp_15.JPG")

# Initialize the object
Img1 = Image(img1)
Img2 = Image(img2)

# Extract the features from each images
kps1, des1 = Img1.findFeatures('sift')
kps2, des2 = Img2.findFeatures('sift')

# Extract the masks to filter the features into several clusters
masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)

kpsCluster1, desCluster1 = Img1.featureCluster(masks1)
kpsCluster2, desCluster2 = Img2.featureCluster(masks2)

# Match the features with corresponding clusters in each image
matches = utils.clusterMatch(desCluster1,desCluster2)
print("Number of original matches: ", len(matches[0])+len(matches[1]))

img_match = utils.drawMatch(Img1,kpsCluster1,Img2,kpsCluster2,matches,draw_params)


# Visualie the matching
plt.figure(1)
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.axis('off')


kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)
homo_mat, inliers_mask = utils.findHomography(matches, kps1_filter, kps2_filter)
matches_inliers = list(itertools.compress(matches, inliers_mask))
img_inliers = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers,None,**draw_params)


plt.figure(2)
plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
plt.axis('off')

plt.show()



    


