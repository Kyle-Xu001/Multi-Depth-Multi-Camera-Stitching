import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import utils
from feature_extraction_test import Image
import feature_matching_test as match_utils

if __name__ == '__main__':
    
    # load the json parameters
    with open('ROI.json','r',encoding='utf8')as fp:
        ROIs = json.load(fp)
    
    # Transform the ROIs from json file
    ROIs1 = np.array(ROIs['lamp17-lamp16']['ROIs1']).reshape(-1,4)
    ROIs2 = np.array(ROIs['lamp17-lamp16']['ROIs2']).reshape(-1,4)
    
    # Load the distorted images
    img1 = cv.imread("dataset/origin_images/lamp_17_distorted_empty_for16.PNG")
    img2 = cv.imread("dataset/origin_images/lamp_16_distorted_empty_for17.PNG")
    
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
    
    # Visualize the feature extraction in each region
    for i in range(len(ROIs1)):
        plt.figure(0)
        plt.subplot(len(ROIs1),2,2*i+1)
        img_kps = cv.drawKeypoints(Img1.img, kpsCluster1[i], None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        plt.imshow(img_kps)
        plt.axis('off')
        
        plt.subplot(len(ROIs1),2,2*i+2)
        img_kps = cv.drawKeypoints(Img2.img, kpsCluster2[i], None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        plt.imshow(img_kps)
        plt.axis('off')
        
    plt.show()