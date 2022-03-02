import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def featureMatch(des1, des2, method, knn=False):
    if method == 'sift' and knn == False:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    elif method == 'sift' and knn == True:
        bf = cv.BFMatcher()
    elif method == 'brisk'and knn == False:
        bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    elif method == 'brisk'and knn == True:
        bf = cv.BFMatcher()
    
    
    if knn == False:
        matches = bf.match(des1, des2)
        matches_good = sorted(matches, key=lambda x: x.distance)
    else:
        matches = bf.knnMatch(des1, des2, k=2)
        
        matches_good = []
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                matches_good.append(m)
    return matches_good


def clusterMatch(desCluster1, desCluster2):
    matches = []
    for i in range(len(desCluster1)):
        des1 = desCluster1[i]
        print("des1: ", len(des1))
        des2 = desCluster2[i]

        bf = cv.BFMatcher()
        
        match = bf.knnMatch(des1, des2, k=2)
        
        matchFilter = []
        for m,n in match:
            if m.distance < 0.8 * n.distance:
                matchFilter.append(m)
        matches.append(matchFilter)
    return matches


def findFeatures(img, method=None):

    if method == 'sift':
        descriptor = cv.SIFT_create()
    elif method == 'brisk':
        descriptor = cv.BRISK_create()
    elif method == 'orb':
        descriptor = cv.ORB_create(nfeatures=5000,nlevels=8)
    
    kps, des = descriptor.detectAndCompute(img, None)

    return kps, des


def equalizeHist(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    img_histequal = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    return img_histequal

def equalizeHist_old(img):
    img_histequal_ = np.zeros(img.shape, dtype=np.uint8)
    img_histequal_[:, :, 0] = cv.equalizeHist(img[:, :, 0])
    img_histequal_[:, :, 1] = cv.equalizeHist(img[:, :, 1])
    img_histequal_[:, :, 2] = cv.equalizeHist(img[:, :, 2])

    return img_histequal_


def getMaskPointsInROIs(kps,ROIs):
    """
    Parameters
    ----------
    pts : n x 2 ndarray
        Each row is a point (x,y)
    ROIs : List of ROIs, each ROI is a size 4 iterable
        Each ROI consists of (x1,y1,x2,y2), where (x1,y1) is the top left point
        and (x2,y2) is the bottom right point

    Returns
    -------
    ndarray of mask

    """
    pts = cv.KeyPoint_convert(kps)
    submasks = []
    for ROI in ROIs:
        x_mask = np.logical_and(pts[:,0] >= ROI[0],pts[:,0] <= ROI[2])
        y_mask = np.logical_and(pts[:,1] >= ROI[1],pts[:,1] <= ROI[3])
        submasks.append(np.logical_and(x_mask,y_mask))
    
    # final_mask = np.zeros(submasks[0].shape,dtype=bool)
    # for mask in submasks:
    #     final_mask = np.logical_or(final_mask,mask)
        
    return submasks

def drawPoints(img,pts):
    """
    Parameters
    ----------
    img : Image ndarray
        
    pts : n by 2 ndarray

    Returns
    -------
    img : Image ndarray
    """
    assert(pts.shape[1]==2)
    n_points = pts.shape[0]
    
    for i in range(n_points):
        pt = tuple(pts[i,:])
        img = cv.circle(img,pt,radius=4,color=(0,255,0))
    
    return img