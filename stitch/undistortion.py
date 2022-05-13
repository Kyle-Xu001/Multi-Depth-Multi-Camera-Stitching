import os
import csv
import glob
import pickle

import numpy as np
import cv2 as cv
from time import *

# Load the Intrinsic Parameter for Camera
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

    return calib_params_all, center_offsets, balance, right_extend


# Define the undistortion function to generate map1 & map2
def calculate_map(img,DIM,mtx,dist,balance,center_offset=0,right_extend=0,map1 = None,map2 = None):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    dim2 = dim1

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


'''Transform the features from original distorted image to the undistorted image'''
def feature_map(map1, map2, features):
    # Define the pts list in undistorted image 
    pts_undistorted = np.zeros_like(features)
    num = [] # which means the number of potential position each feature has
    invalid_index = [] # define the features that should be ignored
        
    # Calculate the time (to be more efficient)
    begin_time = time()
    
    for index,feature in enumerate(features):
        map_x_mask = np.where(np.abs(map1-feature[0])<=0.6,True,False)
        map_y_mask = np.where(np.abs(map2-feature[1])<=0.6,True,False)
        map_mask = np.logical_and(map_x_mask,map_y_mask)
        undistort_list = np.where(map_mask) # potiential positions for one feature
        
        '''
        Estimated Method: Estimate where should feature points be 
                        in the undistorted image based on the mean value
        '''
        if len(undistort_list[0]) == 0:
            x = 0
            y = 0
            invalid_index.append(index)
        else:
            x = np.mean(undistort_list[1])
            y = np.mean(undistort_list[0])
            if x==0 or y==0:
                invalid_index.append(index)
            
        # Update the undistorted coordination
        pts_undistorted[index,:] = [x,y]
        num.append(np.sum(map_mask!=0))
        
        
    end_time = time()
    print('\n\nProcessing Time for Feature Transformation: ',end_time-begin_time)
    
    # Transfer the pts into kps strucutre
    features_undistorted = cv.KeyPoint_convert(pts_undistorted)
    
    print('Number Verification: ',len(features)==len(num))
    print('Maximum Position Selections for one feaure: ',np.max(num))
    print('Average Position Selections for one feaure: ',np.average(num))
    print('Number of Valid Features after Transformation: ',np.count_nonzero(num),'/',len(features))

    return features_undistorted, invalid_index