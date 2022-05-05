import cv2 as cv
import numpy as np


def getPos(ID, pt, trans_params):
    '''get the global position of point from individual camera'''
    # Get the Translation Parameters for specific Lamp ID
    trans_param = np.array(trans_params[ID]["param"]).reshape(-1, 3)
    trans_compensate = np.array(trans_params[ID]["transition"])
    
    # Change the shape of position of the point
    pos = np.array(pt,dtype='float64').reshape([1,1,2])
    
    # Transform the local position to global position
    pos_new = cv.perspectiveTransform(pos, trans_param).flatten()
    
    pt_new = (pos_new[0], pos_new[1])
    pt_new = pt_new - trans_compensate
    return pt_new


def getPos_box(ID, obb, trans_params):
    '''transform the single box to the panorama image'''
    # Get the Translation Parameters for specific Lamp ID
    trans_param = trans_params[ID]["param"]
    trans_compensate = np.array(trans_params[ID]["transition"])
    
    # Change the shape of position of the point
    obb = np.asarray(obb,dtype='float64')
    obb = obb.reshape([1,int(len(obb)/2),2])
    
    # Transform the local position to global position
    obb_transform = cv.perspectiveTransform(obb, trans_param)
    obb_transform = obb_transform.reshape(obb.shape[1:])
    obb_transform = obb_transform - trans_compensate
    return obb_transform


def getPos_box_array(ID, obb, trans_params):
    '''transform the single box to the panorama image'''
    # Get the Translation Parameters for specific Lamp ID
    trans_param = np.array(trans_params[ID]["param"]).reshape(-1, 3)
    trans_compensate = np.array(trans_params[ID]["transition"])
    
    # Change the shape of position of the point
    obb = np.asarray(obb,dtype='float64')
    obb = obb.reshape(-1,2)
    obb = obb.reshape([1,len(obb),2])
    
    # Transform the local position to global position
    obb_transform = cv.perspectiveTransform(obb, trans_param)
    obb_transform = obb_transform.reshape(obb.shape[1:])
    
    obb_transform = obb_transform - trans_compensate
    # Floor Projection
    #obb_transform = projectToFloor_box(obb_transform)
    return obb_transform


def projectToFloor_box(box):
        
        width = 1376
        height = 768
        
        xintercept = 0.07885366*width
        xcoef = -0.16325128
        yintercept = 0.08407642*height
        ycoef = -0.15205884
        
        test = np.copy(box)
        test[:,0] = box[:,0] + xintercept + xcoef*box[:,0]
        test[:,1] = box[:,1] + yintercept + ycoef*box[:,1]

        
        return test