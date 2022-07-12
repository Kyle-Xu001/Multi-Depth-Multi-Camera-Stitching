import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from stitch import Stitch, utils, undistort, feature_map, simpleStitch
from apap_utils import final_size, Apap, get_mesh, get_vertice, uniform_blend, simple_stitch


if __name__ == '__main__':
    # Define the Operating Camera# Define the Operating Camera
    # lamp_id1 = 'lamp01'
    # lamp_id2 = 'lamp02'
    
    # # Load the Original Distorted Images
    # img1_ = cv.imread("dataset/Arie/lamp_01.PNG")
    # img2_ = cv.imread("dataset/Arie/lamp_02.PNG")
    
    
    # '''Calibrate the Original Image for Undistortion'''
    # # Enter the direction of the parameters
    # calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Arie"
    
    # # Calculate the mapping matrix
    # img_undistort1,map1_1,map2_1 = undistort(img1_,lamp_id1,calib_dir)
    # img_undistort2,map1_2,map2_2 = undistort(img2_,lamp_id2,calib_dir)
    

    # '''Select Range of Interest for each Image'''
    # # Define the draw parameters for matching visualization
    # draw_params = dict(matchColor = (0,255,0),
    #                 singlePointColor = (0,0,255),
    #                 flags = cv.DrawMatchesFlags_DEFAULT)

    # # load the matching images
    # img1 = img1_
    # img2 = img2_

    # # Manually define the ROI to locate the area for corresponding images
    # ROIs1 = cv.selectROIs("select the area", img1)
    # ROIs2 = cv.selectROIs("select the area", img2)

    # for i in range(len(ROIs1)):
    #     ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
    #     ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
    #     ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
    #     ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]
    
    # '''Matching Features within Corresponding Areas'''
    # # Initialize the object
    # stitches = Stitch(img1, img2)
    # stitches.featureExtract(ROIs1, ROIs2)
    
    # # Define the matches based on two images
    # matches_list = stitches.clusterMatch('sift', knn=True)    
    
    # # Combine the features in one lists from each cluster
    # stitches.featureIntegrate(matches_list)

    
    # '''Mapping the Original Features into Undistortion Image'''
    # # Filter the invalid matches and transform the features
    # pts1 = cv.KeyPoint_convert(stitches.Img1.kps)
    # pts2 = cv.KeyPoint_convert(stitches.Img2.kps)
    
    # features1, invalid_index1 = feature_map(map1_1, map2_1, pts1)
    # features2, invalid_index2 = feature_map(map1_2, map2_2, pts2)
    
    # matches = utils.matchFilter(stitches.matches, invalid_index1, invalid_index2)    
    
    # '''
    # Find the parameters for homography matrix
    # '''
    # # Calculate the homography matrix for image transformation
    # homo_mat, inliers_mask = utils.findHomography(matches, features1, features2)
    # matches_inliers = list(itertools.compress(matches, inliers_mask))
    
    # img_stitch = simpleStitch(img_undistort1, img_undistort2, homo_mat)

    # plt.figure(1)
    # plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    # plt.axis('off')    
    
    img1 = cv.imread("test_image1.png")
    img2 = cv.imread("test_image2.png")

    X1 = np.load("test_image1_match.npy")
    X2 = np.load("test_image2_match.npy")

    homo_mat = np.load("test_homo_mat.npy")
    X1_ok = np.load("img1_inliers.npy")
    X2_ok = np.load("img2_inliers.npy")

    
    
    mesh_size = 50
    
    # Define the shape of undistortion images
    ori_h, ori_w, _ = img1.shape
    dst_h, dst_w, _ = img2.shape
    
    # Transfer the inlier points into np.array
    final_src = X1_ok.T[:,:2]
    final_dst = X2_ok.T[:,:2]

    final_w, final_h, offset_x, offset_y = final_size(img2, img1, homo_mat)
    
    mesh = get_mesh((final_w, final_h), mesh_size + 1)
    vertices = get_vertice((final_w, final_h), mesh_size, (offset_x, offset_y))
    
    stitcher = Apap([final_w, final_h], [offset_x, offset_y])
    
    # local homography estimating
    local_homography, local_weight = stitcher.local_homography(final_dst, final_src, vertices)
    
    # local warping
    warped_img = stitcher.local_warp(img2, local_homography, mesh)

    # another image pixel move
    dst_temp = np.zeros_like(warped_img)
    dst_temp[offset_y: dst_h + offset_y, offset_x: dst_w + offset_x, :] = img1
    
    result = uniform_blend(warped_img, dst_temp)
    #result = simple_stitch(warped_img,dst_temp)
    
    plt.show()
    cv.imshow("result",warped_img)
    cv.imshow("result1",dst_temp)
    cv.imshow("final_apap",result)
    cv.waitKey(0)
    
    print(final_w, final_h, offset_x, offset_y)
    # img_inliers = cv.drawMatches(img_undistort1,features1,img_undistort2,features2,matches_inliers,None,**draw_params)

    # print("\nNumber of inlier matches: ", len(matches_inliers),"\n")