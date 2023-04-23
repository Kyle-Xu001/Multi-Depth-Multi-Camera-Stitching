import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from stitch import Stitch, utils, undistort, feature_map, simpleStitch

    
if __name__ =='__main__':
    # Define the Operating Camera
    lamp_id1 = 'lamp02'
    lamp_id2 = 'lamp01'
    
    # Load the Original Distorted Images
    img1_ = cv.imread("dataset/Arie/lamp_02.PNG")
    img2_ = cv.imread("dataset/Arie/lamp_01.PNG")
    
    # The stitch order need to be flipped to be keep the right image unchanged
    #img1_ = cv.flip(img1_, 0)
    #img2_ = cv.flip(img2_, 0)

    '''Calibrate the Original Image for Undistortion'''
    # Enter the direction of the parameters
    calib_dir = "params/calib_params_Arie"
    
    # Calculate the mapping matrix
    img_undistort1,map1_1,map2_1 = undistort(img1_,lamp_id1,calib_dir)
    img_undistort2,map1_2,map2_2 = undistort(img2_,lamp_id2,calib_dir)
    
    # Visualize the undistortion process
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(cv.cvtColor(img1_, cv.COLOR_BGR2RGB))
    plt.title('(a) Original Distorted Image [%s]'%(lamp_id1))
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.imshow(cv.cvtColor(img_undistort1, cv.COLOR_BGR2RGB))
    plt.title('(b) Undistorted Image [%s]'%(lamp_id1))
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.imshow(cv.cvtColor(img2_, cv.COLOR_BGR2RGB))
    plt.title('(c) Original Distorted Image [%s]'%(lamp_id2))
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.imshow(cv.cvtColor(img_undistort2, cv.COLOR_BGR2RGB))
    plt.title('(d) Undistorted Image [%s]'%(lamp_id2))
    plt.axis('off')
    
    
    '''Select Range of Interest for each Image'''
    # Define the draw parameters for matching visualization
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (0,0,255),
                    flags = cv.DrawMatchesFlags_DEFAULT)

    # load the matching images
    img1 = img1_
    img2 = img2_

    # Manually define the ROI to locate the area for corresponding images
    ROIs1 = cv.selectROIs("select the area", img1)
    ROIs2 = cv.selectROIs("select the area", img2)

    for i in range(len(ROIs1)):
        ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
        ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
        ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
        ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]
    
    '''Matching Features within Corresponding Areas'''
    # Initialize the object
    stitches = Stitch(img1, img2)
    stitches.featureExtract(ROIs1, ROIs2)
    
    # Define the matches based on two images
    matches_list = stitches.clusterMatch('sift', knn=True)    

    # Show the number of matches
    matchNum = 0
    for i in range(len(matches_list)):
        matchNum += len(matches_list[i])
        print("-- Number of original matches in area (%d): %d"%(i, len(matches_list[i])))
    print("Number of original total matches: ", matchNum)

    # draw the matches in each cluster
    utils.drawMatch(stitches.Img1,stitches.Img2,matches_list,draw_params)
    
    # Combine the features in one lists from each cluster
    stitches.featureIntegrate(matches_list)

    
    '''Mapping the Original Features into Undistortion Image'''
    # Filter the invalid matches and transform the features
    pts1 = cv.KeyPoint_convert(stitches.Img1.kps)
    pts2 = cv.KeyPoint_convert(stitches.Img2.kps)
    
    features1, invalid_index1 = feature_map(map1_1, map2_1, pts1)
    features2, invalid_index2 = feature_map(map1_2, map2_2, pts2)
    
    matches = utils.matchFilter(stitches.matches, invalid_index1, invalid_index2)    
    
    
    '''
    Find the parameters for homography matrix
    '''
    # Calculate the homography matrix for image transformation
    homo_mat, inliers_mask = utils.findHomography(matches, features1, features2)
    matches_inliers = list(itertools.compress(matches, inliers_mask))
    img_inliers = cv.drawMatches(img_undistort1,features1,img_undistort2,features2,matches_inliers,None,**draw_params)

    print("\nNumber of inlier matches: ", len(matches_inliers),"\n")


    plt.figure(2)
    plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
    plt.title("Inlier Matches for Total Selected Area")
    plt.axis('off')
    
    
    
    '''
    Stitch the Images
    '''
    img_stitch = simpleStitch(img_undistort1, img_undistort2, homo_mat)

    plt.figure(3)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    
    '''
    Print the parameters of homography matrix
    '''
    np.set_printoptions(suppress=True)
    print(homo_mat.flatten().tolist())