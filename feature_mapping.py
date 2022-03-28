from time import *
import numpy as np
import cv2 as cv
import math


def feature_map(map1, map2, features):
    # Define the pts list in undistorted image 
    pts_undistorted = np.zeros_like(features)
    num = [] # which means the number of potential position each feature has
    invalid_index = [] # define the features that should be ignored
    
    #features = features[:100]
    
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