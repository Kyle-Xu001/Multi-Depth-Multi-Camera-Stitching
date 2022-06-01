import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from stitch import Stitch, utils, undistort, feature_map, simpleStitch
from apap_utils import final_size, Apap, get_mesh, get_vertice, uniform_blend


if __name__ == '__main__':
    
    # Define the Operating Camera# Define the Operating Camera
    lamp_id1 = 'lamp18'
    lamp_id2 = 'lamp17'
    
    # Load the Original Distorted Images
    img1_ = cv.imread("dataset/lamp14151617-lamp18/lamp_18_012313.PNG")
    img2_ = cv.imread("dataset/lamp14151617-lamp18/lamp_17_012313.PNG")
    
    '''Calibrate the Original Image for Undistortion'''
    # Enter the direction of the parameters
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"
    
    # Calculate the mapping matrix
    img_undistort1,map1_1,map2_1 = undistort(img1_,lamp_id1,calib_dir)
    img_undistort2,map1_2,map2_2 = undistort(img2_,lamp_id2,calib_dir)
    
    # Define the global_homography matrix
    homo_mat = [
        1.1386527445204575,
        0.031927187738166204,
        -109.45549680535996,
        0.04498832067998204,
        1.0320031734670985,
        463.08525965607436,
        0.00010009854442945858,
        0.00006315329701074331,
        1.0
    ]
    homo_mat = np.array(homo_mat).reshape(-1, 3)
    print(homo_mat)

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
    
    img_stitch = simpleStitch(img_undistort1, img_undistort2, homo_mat)

    plt.figure(1)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # Define the shape of undistortion images
    ori_h, ori_w, _ = img_undistort2.shape
    dst_h, dst_w, _ = img_undistort1.shape
    
    # Transfer the inlier points into np.array
    final_src = []
    final_dst = []
    for match in matches_inliers:
        trainIdx = match.trainIdx
        queryIdx = match.queryIdx
        final_dst.append(features1[queryIdx].pt)
        final_src.append(features2[trainIdx].pt)
    final_src = np.array(final_src)
    final_dst = np.array(final_dst)

    final_w, final_h, offset_x, offset_y = final_size(img_undistort2, img_undistort1, homo_mat)
    
    mesh = get_mesh((final_w, final_h), mesh_size + 1)
    vertices = get_vertice((final_w, final_h), mesh_size, (offset_x, offset_y))
    
    stitcher = Apap([final_w, final_h], [offset_x, offset_y])
    
    # local homography estimating
    local_homography, local_weight = stitcher.local_homography(final_src, final_dst, vertices)
    
    # local warping
    warped_img = stitcher.local_warp(img_undistort2, local_homography, mesh)

    # another image pixel move
    dst_temp = np.zeros_like(warped_img)
    dst_temp[offset_y: dst_h + offset_y, offset_x: dst_w + offset_x, :] = img_undistort1
    result = uniform_blend(warped_img, dst_temp)
    
    cv.imshow("result",warped_img)
    cv.imshow("result1",dst_temp)
    cv.imshow("final_apap",result)
    cv.waitKey(0)
    
    plt.show()
    print(final_w, final_h, offset_x, offset_y)
    # img_inliers = cv.drawMatches(img_undistort1,features1,img_undistort2,features2,matches_inliers,None,**draw_params)

    # print("\nNumber of inlier matches: ", len(matches_inliers),"\n")