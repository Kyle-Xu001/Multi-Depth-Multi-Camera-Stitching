import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from stitch import Stitch, utils, undistort, feature_map, simpleStitch, alphablend


if __name__ == '__main__':
    
    # Define the Operating Camera
    lamp_id1 = 'lamp01'
    lamp_id2 = 'lamp02'
    
    # Load the Original Distorted Images
    img1_ = cv.imread("dataset/Arie/lamp_01.PNG")
    img2_ = cv.imread("dataset/Arie/lamp_02.PNG")
    
    
    '''Calibrate the Original Image for Undistortion'''
    
    # Enter the direction of the parameters
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Arie"
    
    # Calculate the mapping matrix
    img_undistort1,map1_1,map2_1 = undistort(img1_,lamp_id1,calib_dir)
    img_undistort2,map1_2,map2_2 = undistort(img2_,lamp_id2,calib_dir)
    
    img1 = cv.rotate(img_undistort1, cv.ROTATE_90_CLOCKWISE)
    img2 = cv.rotate(img_undistort2, cv.ROTATE_90_CLOCKWISE)
    
    
    '''Select Range of Interest for each Image'''
    # Define the draw parameters for matching visualization
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (0,0,255),
                    flags = cv.DrawMatchesFlags_DEFAULT)

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
    
    # Combine the features in one lists from each cluster
    stitches.featureIntegrate(matches_list)
    
    # Load points from the matches
    srcpoints = np.float32([stitches.Img1.kps[m.queryIdx].pt for m in stitches.matches])
    dstpoints = np.float32([stitches.Img2.kps[m.trainIdx].pt for m in stitches.matches])
    
    fp = np.vstack((srcpoints.T, np.ones((1, len(stitches.matches)))))
    tp = np.vstack((dstpoints.T, np.ones((1, len(stitches.matches)))))
    
    # Save the data in the images
    cv.imwrite("test_image1.png",img1)
    cv.imwrite("test_image2.png",img2)
    np.save("test_image1_match.npy",fp)
    np.save("test_image2_match.npy",tp)
    
    
    '''
    Find the parameters for homography matrix
    '''
    # Calculate the homography matrix for image transformation
    homo_mat, inliers_mask = utils.findHomography(stitches.matches, stitches.Img1.kps, stitches.Img2.kps)
    matches_inliers = list(itertools.compress(stitches.matches, inliers_mask))
    
    # Load points from the matches
    srcpoints = np.float32([stitches.Img1.kps[m.queryIdx].pt for m in matches_inliers])
    dstpoints = np.float32([stitches.Img2.kps[m.trainIdx].pt for m in matches_inliers])
    
    X1_ok = np.vstack((srcpoints.T, np.ones((1, len(matches_inliers)))))
    X2_ok = np.vstack((dstpoints.T, np.ones((1, len(matches_inliers)))))
    
    img_stitch = alphablend(img1, img2, homo_mat)
    
    np.save("test_homo_mat.npy",homo_mat)
    np.save("img1_inliers.npy",X1_ok)
    np.save("img2_inliers.npy",X2_ok)
    

    plt.figure(1)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')    
    plt.show()