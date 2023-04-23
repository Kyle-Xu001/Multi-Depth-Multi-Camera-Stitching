import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from stitch import utils,Stitch
import stitch

if __name__ == '__main__':
    '''This script will be tested for multiple ROIs'''
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Image Stitching")
    parser.add_argument("--img1", type=str)
    parser.add_argument("--img2", type=str)
    parser.add_argument("--rotate", action="store_true", help="Rotate the image to get better visualization")
    args, _ = parser.parse_known_args()
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    
    # load the matching images
    img1 = cv.imread(args.img1)
    img2 = cv.imread(args.img2)
    # img1 = cv.imread("dataset/Arie/lamp_02_Arie.PNG")
    # img2 = cv.imread("dataset/Arie/lamp_01_Arie.PNG")
    # img1 = cv.imread("dataset/example_image/APAP-railtracks/1.JPG")
    # img2 = cv.imread("dataset/example_image/APAP-railtracks/2.JPG")
    # img1 = cv.imread("dataset/example_image/NISwGSP-denny/denny02.jpg")
    # img2 = cv.imread("dataset/example_image/NISwGSP-denny/denny03.jpg")

    if args.rotate:
        img1 = np.rot90(img1,1) 
        img2 = np.rot90(img2,1)
    
    # Manually define the ROI to locate the area for corresponding images
    cv.namedWindow("Area Selection", cv.WINDOW_NORMAL)
    cv.resizeWindow("Area Selection", 800, 600)
    ROIs1 = cv.selectROIs("Area Selection", img1)
    ROIs2 = cv.selectROIs("Area Selection", img2)

    for i in range(len(ROIs1)):
        ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
        ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
        ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
        ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]
        
    
    '''SIFT MATCHING WITHOUT KNN'''
    # Initialize the Stitch Class
    stitch_sift = Stitch(img1, img2)
    stitch_sift.featureExtract(ROIs1, ROIs2)
    
    # Define the matches based on two images
    matches_list = stitch_sift.clusterMatch('sift', knn=False)
    
    # Combine the features in one lists from each cluster
    stitch_sift.featureIntegrate(matches_list)
    _, matches_inliers_sift = stitch_sift.homoEstimate()
    
    
    '''SIFT MATCHING WITH KNN'''
    # Initialize the Stitch Class
    stitch_sift_knn = Stitch(img1, img2)
    stitch_sift_knn.featureExtract(ROIs1, ROIs2)
    
    # Define the matches based on two images
    matches_list = stitch_sift_knn.clusterMatch('sift', knn=True)
    
    # draw the matches in each cluster
    utils.drawMatch(stitch_sift.Img1,stitch_sift.Img2,matches_list,draw_params)
    
    # Combine the features in one lists from each cluster
    stitch_sift_knn.featureIntegrate(matches_list)
    _, matches_inliers_sift_knn = stitch_sift_knn.homoEstimate()
    
    
    '''BRISK MATCHING WITHOUT KNN'''
    # Initialize the Stitch Class
    stitch_brisk = Stitch(img1, img2)
    
    # Update the image features
    stitch_brisk.Img1.findFeatures('brisk')
    stitch_brisk.Img2.findFeatures('brisk')
    stitch_brisk.featureExtract(ROIs1, ROIs2)
    
    # Define the matches based on two images
    matches_list = stitch_brisk.clusterMatch('brisk', knn=False)
    
    # Combine the features in one lists from each cluster
    stitch_brisk.featureIntegrate(matches_list)
    _, matches_inliers_brisk = stitch_brisk.homoEstimate()

    '''BRISK MATCHING WITH KNN'''
    # Initialize the Stitch Class
    stitch_brisk_knn = Stitch(img1, img2)
    
    # Update the image features
    stitch_brisk_knn.Img1.findFeatures('brisk')
    stitch_brisk_knn.Img2.findFeatures('brisk')
    stitch_brisk_knn.featureExtract(ROIs1, ROIs2)
    
    # Define the matches based on two images
    matches_list = stitch_brisk_knn.clusterMatch('brisk', knn=True)
    
    # Combine the features in one lists from each cluster
    stitch_brisk_knn.featureIntegrate(matches_list)
    _, matches_inliers_brisk_knn = stitch_brisk_knn.homoEstimate()
    
    
    '''Draw the Inliers'''
    img_inliners_sift = cv.drawMatches(stitch_sift.Img1.img, stitch_sift.Img1.kps, stitch_sift.Img2.img, stitch_sift.Img2.kps, matches_inliers_sift, None, **draw_params)
    img_inliners_sift_knn = cv.drawMatches(stitch_sift_knn.Img1.img, stitch_sift_knn.Img1.kps, stitch_sift_knn.Img2.img, stitch_sift_knn.Img2.kps, matches_inliers_sift_knn, None, **draw_params)
    img_inliners_brisk = cv.drawMatches(stitch_brisk.Img1.img, stitch_brisk.Img1.kps, stitch_brisk.Img2.img, stitch_brisk.Img2.kps, matches_inliers_brisk, None, **draw_params)
    img_inliners_brisk_knn = cv.drawMatches(stitch_brisk_knn.Img1.img, stitch_brisk_knn.Img1.kps, stitch_brisk_knn.Img2.img, stitch_brisk_knn.Img2.kps, matches_inliers_brisk_knn, None, **draw_params)  
    
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.title("Brute Force Matching on SIFT Features\n(# of Inliners: %d)" %(len(matches_inliers_sift)))
    plt.imshow(cv.cvtColor(img_inliners_sift, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.title("Brute Force KNN Matching on SIFT Features\n(# of Inliners: %d)" %(len(matches_inliers_sift_knn)))
    plt.imshow(cv.cvtColor(img_inliners_sift_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.title("Brute Force Matching on BRISK Features\n(# of Inliners: %d)" %(len(matches_inliers_brisk)))
    plt.imshow(cv.cvtColor(img_inliners_brisk, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.title("Brute Force KNN Matching on BRISK Features\n(# of Inliners: %d)" %(len(matches_inliers_brisk_knn)))
    plt.imshow(cv.cvtColor(img_inliners_brisk_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

    plt.axis('off')