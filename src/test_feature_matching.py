import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import itertools
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from stitch import Image, utils


if __name__ == '__main__':
    '''This script will be tested for feature matching'''
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
    #img1 = cv.imread("dataset/example_image/APAP-railtracks/1.JPG")
    #img2 = cv.imread("dataset/example_image/APAP-railtracks/2.JPG")
    #img1 = cv.imread("dataset/example_image/NISwGSP-denny/denny02.jpg")
    #img2 = cv.imread("dataset/example_image/NISwGSP-denny/denny03.jpg")

    if args.rotate:
        img1 = np.rot90(img1,1)
        img2 = np.rot90(img2,1)
    
    # Initialize the Stitch Class
    Img1 = Image(img1)
    Img2 = Image(img2)       
    
    '''SIFT Features Matching Comparison'''
    # Find SIFT features of matching images
    kps1_filter, des1_filter = Img1.kps, Img1.des
    kps2_filter, des2_filter = Img2.kps, Img2.des
    

    # BFMatches(des1_filter, des2_filter)
    matches_sift = utils.featureMatch(des1_filter, des2_filter, 'sift')
    matches_sift_knn = utils.featureMatch(des1_filter, des2_filter, 'sift', knn=True)
    

    img_sift = cv.drawMatches(Img1.img, kps1_filter, Img2.img, kps2_filter, matches_sift[:300],None,**draw_params)
    img_sift_knn = cv.drawMatches(Img1.img, kps1_filter, Img2.img, kps2_filter, matches_sift_knn[:300],None,**draw_params)
    
    '''BRISK Features Matching Comparison'''
    # Extract the BRISK features
    kps1_filter_, des1_filter_ = Img1.findFeatures('brisk')
    kps2_filter_, des2_filter_ = Img2.findFeatures('brisk')
    
    # BFMatches(des1_filter, des2_filter)
    matches_brisk = utils.featureMatch(des1_filter_, des2_filter_, 'brisk')
    matches_brisk_knn = utils.featureMatch(des1_filter_, des2_filter_, 'brisk',knn=True)

    img_brisk = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_brisk[:300],None,**draw_params)
    img_brisk_knn = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_brisk_knn,None,**draw_params)
    
    
    '''Visualize the Feature Matching Result'''
    fig1 = plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1)
    plt.title("Brute Force Matching on SIFT Features\n(# of Matches: %d)" %(len(matches_sift)))
    plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.title("Brute Force KNN Matching on SIFT Features\n(# of Matches: %d)" %(len(matches_sift_knn)))
    plt.imshow(cv.cvtColor(img_sift_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.title("Brute Force Matching on BRISK Features\n(# of Matches: %d)" %(len(matches_brisk)))
    plt.imshow(cv.cvtColor(img_brisk, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.title("Brute Force KNN Matching on BRISK Features\n(# of Matches: %d)" %(len(matches_brisk_knn)))
    plt.imshow(cv.cvtColor(img_brisk_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    fig1.tight_layout()
    plt.show()
    

    '''Estimate the Inliers'''
    homo_mat_sift, inliers_mask_sift = utils.findHomography(matches_sift,kps1_filter,kps2_filter)
    homo_mat_sift_knn, inliers_mask_sift_knn = utils.findHomography(matches_sift_knn,kps1_filter,kps2_filter)
    homo_mat_brisk, inliers_mask_brisk = utils.findHomography(matches_brisk,kps1_filter_,kps2_filter_)
    homo_mat_brisk_knn, inliers_mask_brisk_knn = utils.findHomography(matches_brisk_knn,kps1_filter_,kps2_filter_)
    
    matches_inliers_sift = list(itertools.compress(matches_sift, inliers_mask_sift))
    matches_inliers_sift_knn = list(itertools.compress(matches_sift_knn, inliers_mask_sift_knn))
    matches_inliers_brisk = list(itertools.compress(matches_brisk, inliers_mask_brisk))
    matches_inliers_brisk_knn = list(itertools.compress(matches_brisk_knn, inliers_mask_brisk_knn))
    
    '''Draw the Inliers'''
    img_inliners_sift = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers_sift,None,**draw_params)
    img_inliners_sift_knn = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers_sift_knn,None,**draw_params)
    img_inliners_brisk = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_inliers_brisk,None,**draw_params)
    img_inliners_brisk_knn = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_inliers_brisk_knn,None,**draw_params)    
    
    fig2 = plt.figure(figsize=(15, 10))
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
    
    fig2.tight_layout()
    plt.show()