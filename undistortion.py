import os
import csv
import glob
import pickle
import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from image_feature_extraction_test import Image
import image_match_test as match_utils
import feature_mapping


# Load the intrinsic parameter for every lamp
def load_params(calib_dir):
    balance = 0
    right_extend = 300
    
    calib_params_all = dict()
    calib_params_paths = glob.glob(os.path.join(calib_dir,'*.pkl'))
    for calib_params_path in calib_params_paths:
        with open(calib_params_path,'rb') as f:
            lamp_id = os.path.basename(calib_params_path)[:6]
            calib_params_all[lamp_id] = pickle.load(f)

    center_offsets = dict()
    center_offsets_path = os.path.join(calib_dir,'center_offsets.csv')
    with open(center_offsets_path) as f:
        reader = csv.reader(f,delimiter=' ')
        for row in reader:
            lamp_id,center_offset = row
            center_offsets[lamp_id] = int(center_offset)

    return calib_params_all,center_offsets,balance,right_extend

# Define the undistortion function to generate map1 & map2
def calculate_map(img,DIM,mtx,dist,balance,center_offset=0,right_extend=0,map1 = None, map2 = None):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    #assert dim1[0]/dim1[1] == DIM[0]/DIM[1]
    dim2 = dim1
    # dim3 = dim1
    dim3 = (dim2[0]+right_extend,dim2[1])
    scaled_K = mtx * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0
    new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, dist, dim2, np.eye(3), balance=balance)
    new_K[0,2] += center_offset
    map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, dist, np.eye(3), new_K, dim3, cv.CV_32FC1)

    return map1,map2

# Generate the undistorted image and corresponding mapping function
def undistort(img,lamp_id,calib_dir,map1 = None, map2 = None):
    # Load the calibration
    calib_params_all,center_offsets,balance,right_extend = load_params(calib_dir)
    
    DIM, mtx, dist = calib_params_all[lamp_id]
    center_offset = center_offsets[lamp_id]
    img_width = int(img.shape[1])
    img_height = int(img.shape[0])
    
    scale = img_width/DIM[0]
    right_extend = int(right_extend*scale)
    center_offset = int(center_offset*scale)
    map1, map2 = calculate_map(img,DIM,mtx,dist,balance,center_offset,right_extend)
    undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return undistorted_img,map1,map2


    
    
    
    

if __name__ =='__main__':
    # Define the operating lamp
    lamp_id1 = 'lamp19'
    lamp_id2 = 'lamp18'
    
    # Load the distorted images
    img1_ = cv.imread("lamp_19_distorted_empty.PNG")
    img2_ = cv.imread("lamp_18_distorted_empty.PNG")
    
    # Enter the direction of the parameters
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"
    
    # Calculate the mapping matrix
    img_undistort1,map1_1,map2_1 = undistort(img1_,lamp_id1,calib_dir)
    img_undistort2,map1_2,map2_2 = undistort(img2_,lamp_id2,calib_dir)
    
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
    
    # Define the draw parameters for matching visualization
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (0,0,255),
                    flags = cv.DrawMatchesFlags_DEFAULT)

    # load the matching images
    #img1 = np.rot90(img1_,1) 
    #img2 = np.rot90(img2_,1)
    img1 = img1_
    img2 = img2_
    
    if (img1.shape[0]>img1.shape[1]):
    # Manually define the ROI to locate the area for corresponding images
        # ROIs1 = [
        #     [450, 950, 768, 1300],
        #     [450, 750, 768, 1000],
        #     [450, 450, 768, 750],
        #     [450, 250, 768, 450]]
        # ROIs2 = [
        #     [0, 950,350, 1300],
        #     [0, 750, 350, 1000],
        #     [0, 500, 350, 800],
        #     [0, 250, 350, 500]]
        ROIs1 = [
            #[450, 850, 768, 1300],
            [450, 625, 768, 900],
            #[450, 300, 768, 625],
            [450, 150, 768, 350]]
        ROIs2 = [
            #[0, 900,350, 1300],
            [0, 700, 350, 950],
            #[0, 450, 350, 750],
            [0, 150, 350, 450]]
    else:
        ROIs1 = [
            [100, 400, 350, 767],
            [300, 400, 500, 767],
            [500, 400, 950, 767],
            [800, 400, 1100, 7/67]]

        ROIs2 = [
            [100, 0, 450, 400],
            [350, 0, 550, 400],
            [550, 0, 1000,400],
            [800, 0, 1150, 400]]
        # ROIs1 = [
        #     [50, 450, 300, 767],
        #     [300, 450, 576, 767],
        #     [576, 450, 876, 767],
        #     [876, 450, 1126, 767]]

        # ROIs2 = [
        #     [0, 0, 250, 400],
        #     [276, 0, 476, 400],
        #     [476, 0, 776, 400],
        #     [776, 0, 1126, 400]]

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

    # Show the number of matches
    matchNum = 0
    for i in range(len(matches)):
        matchNum += len(matches[i])
        print("-- Number of original matches in each area", len(matches[i]))
    print("Number of original total matches: ", matchNum)

    # draw the matches in each cluster
    utils.drawMatch(Img1,kpsCluster1,Img2,kpsCluster2,matches,draw_params)
    
    # Integrate the clusters into one list
    kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)
    
    # Visualize the total matches
    # plt.figure(2)
    # img_match = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches,None,**draw_params)
    # plt.axis('off')
    # plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    # plt.title("Feature Matching for Total Selected Area")
    
    
    pts1 = cv.KeyPoint_convert(kps1_filter)
    pts2 = cv.KeyPoint_convert(kps2_filter)
    features1, invalid_index1 = feature_mapping.feature_map(map1_1, map2_1, pts1)
    features2, invalid_index2 = feature_mapping.feature_map(map1_2, map2_2, pts2)
    matches = utils.matchFilter(matches, invalid_index1, invalid_index2)
    
    # # Visualize the total matches
    plt.figure(2)
    img_match = cv.drawMatches(img_undistort1,features1,img_undistort2,features2,matches,None,**draw_params)
    plt.axis('off')
    plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    plt.title("Feature Matching for Total Selected Area")
    
    '''
    Find the parameters for homography matrix
    '''
    # Calculate the homography matrix for image transformation
    homo_mat, inliers_mask = utils.findHomography(matches, features1, features2)
    matches_inliers = list(itertools.compress(matches, inliers_mask))
    img_inliers = cv.drawMatches(img_undistort1,features1,img_undistort2,features2,matches_inliers,None,**draw_params)

    print("\nNumber of inlier matches: ", len(matches_inliers),"\n")


    plt.figure(3)

    plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
    plt.title("Inlier Matches for Total Selected Area")
    plt.axis('off')
    
    
    '''
    Stitch the Images
    '''
    # Get the position of vertices
    posVerts = utils.transformVerts(img_size=np.array([Img1.img.shape[1],Img1.img.shape[0]]), homo_mat=homo_mat)
    # print("Left Top: ",posVerts[0,:],"\n",
    #       "Right Top: ",posVerts[1,:],"\n",
    #       "Right Bottom: ",posVerts[2,:],"\n",
    #       "Left Bottom: ",posVerts[3,:],"\n")
        
    x_min = posVerts[:,0].min()
    x_max = posVerts[:,0].max()
    y_min = posVerts[:,1].min()
    y_max = posVerts[:,1].max()
    print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %(x_min,x_max,y_min,y_max))

    stitch_size = (x_max,y_max)

    homo_mat_ = np.eye(3)
    img_super = cv.warpPerspective(Img1.img, homo_mat_,stitch_size,borderValue=(0,0,0))
    img_transform = cv.warpPerspective(Img2.img, homo_mat,stitch_size,borderValue=(0,0,0))

    # Combine the image on one super image
    high_y = np.min(posVerts[:,1])
    img_transform[high_y:high_y,:,:] = 0
    img_super[img_transform>0]=0

    img_stitch = img_transform + img_super

    plt.figure(3)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')


    plt.show()