import os
import csv
import glob
import pickle
import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from ImageStitch import Stitch
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
    return undistorted_img, map1, map2


    
    
    
    

if __name__ =='__main__':
    # Define the operating lamp
    lamp_id1 = 'lamp05'
    lamp_id2 = 'lamp06'
    
    # Load the distorted images
    img1_ = cv.imread("dataset/lamp_05_122014_for06.PNG")
    img2_ = cv.imread("dataset/lamp_06_122014.PNG")
    
    #img1_ = cv.flip(img1_, 0)
    #img2_ = cv.flip(img2_, 0)

    # Enter the direction of the parameters
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"
    
    # Calculate the mapping matrix
    img_undistort1,map1_1,map2_1 = undistort(img1_,lamp_id1,calib_dir)
    img_undistort2,map1_2,map2_2 = undistort(img2_,lamp_id2,calib_dir)
    
    # plt.figure(1)q
    # plt.subplot(2,2,1)
    # plt.imshow(cv.cvtColor(img1_, cv.COLOR_BGR2RGB))
    # plt.title('(a) Original Distorted Image [%s]'%(lamp_id1))
    # plt.axis('off')
    # plt.subplot(2,2,2)
    # plt.imshow(cv.cvtColor(img_undistort1, cv.COLOR_BGR2RGB))
    # plt.title('(b) Undistorted Image [%s]'%(lamp_id1))
    # plt.axis('off')
    # plt.subplot(2,2,3)
    # plt.imshow(cv.cvtColor(img2_, cv.COLOR_BGR2RGB))
    # plt.title('(c) Original Distorted Image [%s]'%(lamp_id2))
    # plt.axis('off')
    # plt.subplot(2,2,4)
    # plt.imshow(cv.cvtColor(img_undistort2, cv.COLOR_BGR2RGB))
    # plt.title('(d) Undistorted Image [%s]'%(lamp_id2))
    # plt.axis('off')
    
    # Define the draw parameters for matching visualization
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (0,0,255),
                    flags = cv.DrawMatchesFlags_DEFAULT)

    # load the matching images
    img1 = img1_
    img2 = img2_

    # if (img1.shape[0]>img1.shape[1]):
    # # Manually define the ROI to locate the area for corresponding images
    #     # ROIs1 = [
    #     #     [450, 950, 768, 1300],
    #     #     [450, 750, 768, 1000],
    #     #     [450, 450, 768, 750],
    #     #     [450, 250, 768, 450]]
    #     # ROIs2 = [
    #     #     [0, 950,350, 1300],
    #     #     [0, 750, 350, 1000],
    #     #     [0, 500, 350, 800],
    #     #     [0, 250, 350, 500]]
    #     ROIs1 = [
    #         [425, 750, 767, 1050],
    #         [425, 450, 767, 800],
    #         [425, 250, 767, 500],
    #         [425,  50, 767, 300]]
    #     ROIs2 = [
    #         [0, 800, 350, 1075],
    #         [0, 500, 350, 850],
    #         [0, 300, 350, 550],
    #         [0,   0, 350, 350]]
    # else:
    #     ROIs1 = [
    #         [185, 450, 462, 767],
    #         [630, 450, 900, 767]]

    #     ROIs2 = [
    #         [160, 0, 390, 310],
    #         [575, 0, 850, 310]]
        
        # ROIs1 = [
        #     [250, 425, 530, 767],
        #     [550, 425, 850, 670],
        #     [820, 425, 960, 767]]

        # ROIs2 = [
        #     [220, 0, 400, 300],
        #     [420, 0, 680, 85],
        #     [700, 0, 900, 350]]
    ROIs1 = cv.selectROIs("select the area", img1)
    ROIs2 = cv.selectROIs("select the area", img2)

    for i in range(len(ROIs1)):
        ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
        ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
        ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
        ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]
        
    # Initialize the object
    stitches = Stitch(img1, img2)
    stitches.featureExtract(ROIs1, ROIs2)
    
    # Match the features with corresponding clusters in each image
    homo_mat, matches_inliers = stitches.homoEstimate()
 

    # # Generate the heatmap Image
    # heatImg = utils.featureHeatmap(Img1.img,kps1)
    # heatImg2 = utils.featureHeatmap(Img2.img,kps2)
    # plt.figure(5)
    # plt.subplot(2,1,1)
    # plt.imshow(heatImg,cmap='jet')
    # plt.title('(a) Feature Distribution on Distorted Image [PNG]')
    # plt.axis('off')
    
    # plt.subplot(2,1,2)
    # plt.imshow(heatImg2,cmap='jet')
    # plt.title('(b) Feature Distribution on Undistorted Image [JPG]')
    # plt.axis('off')
    
    
    # plt.show()



    # # Show the number of matches
    # matchNum = 0
    # for i in range(len(matches)):
    #     matchNum += len(matches[i])
    #     print("-- Number of original matches in area (%d): %d"%(i, len(matches[i])))
    # print("Number of original total matches: ", matchNum)

    # # draw the matches in each cluster
    # utils.drawMatch(s,kpsCluster1,Img2,kpsCluster2,matches,draw_params)
    
    # # Integrate the clusters into one list
    # kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)
    # plt.show()
        
    # Filter the invalid matches and transform the features
    pts1 = cv.KeyPoint_convert(stitches.Img1.kps)
    pts2 = cv.KeyPoint_convert(stitches.Img2.kps)
    features1, invalid_index1 = feature_mapping.feature_map(map1_1, map2_1, pts1)
    features2, invalid_index2 = feature_mapping.feature_map(map1_2, map2_2, pts2)
    matches = utils.matchFilter(stitches.matches, invalid_index1, invalid_index2)
    
    
    # Visualize the total matches
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
    posVerts = utils.transformVerts(img_size=np.array([stitches.Img1.img.shape[1],stitches.Img1.img.shape[0]]), homo_mat=homo_mat)
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
    img_super = cv.warpPerspective(img_undistort1, homo_mat_,stitch_size,borderValue=(0,0,0))
    img_transform = cv.warpPerspective(img_undistort2, homo_mat,stitch_size,borderValue=(0,0,0))

    # Combine the image on one super image
    high_y = np.min(posVerts[:,1])
    img_transform[high_y:high_y,:,:] = 0
    img_super[img_transform>0]=0

    img_stitch = img_transform + img_super

    plt.figure(4)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    
    '''
    Print the parameters of homography matrix
    '''
    np.set_printoptions(suppress=True)
    print(homo_mat.flatten().tolist())