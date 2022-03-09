import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import itertools

from image_feature_extraction_test import Image
import utils

def getMaskPointsInROIs(kps,ROIs):
    """
    Parameters
    ----------
    kps : List of feature elements
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
    
    final_mask = np.zeros(submasks[0].shape,dtype=bool)
    for mask in submasks:
        final_mask = np.logical_or(final_mask,mask)
        
    return final_mask

def findHomography(matches, kps1, kps2):
    queryIdxs = [match.queryIdx for match in matches]
    trainIdxs = [match.trainIdx for match in matches]
    kps2 = cv.KeyPoint_convert(kps2)
    kps1 = cv.KeyPoint_convert(kps1)
    homo_mat,inliers_mask = cv.findHomography(kps2[trainIdxs],kps1[queryIdxs],method=cv.RANSAC,ransacReprojThreshold=2)

    return homo_mat, inliers_mask



if __name__ == '__main__':
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    
    # load the matching images
    img1 = cv.imread("lamp_16.JPG")
    img2 = cv.imread("lamp_15.JPG")

    img1 = np.rot90(img1,1) 
    img2 = np.rot90(img2,1)
    
    #img1 = utils.equalizeHist_old(img1)
    #img2 = utils.equalizeHist_old(img2)
    
    # Equalize histogram of images
    Img1 = Image(img1)
    Img2 = Image(img2)
    
    Img1.equalizeHist()
    Img2.equalizeHist()



    # Find SIFT features of matching images
    kps1_sift, dps1_sift = Img1.findFeatures('sift')
    kps2_sift, dps2_sift = Img2.findFeatures('sift')

    # Manual Set
    # ROIs1 = np.array([[160,350,1250,760]])
    # ROIs2 = np.array([[160,0,1250,400]])
    ROIs1 = np.array([[350,150,760,1250]])
    ROIs2 = np.array([[0,150,400,1250]])
    
    final_mask1 = getMaskPointsInROIs(kps1_sift,ROIs1)
    kps1_filter, des1_filter = Img1.featureFilter(final_mask1)

    final_mask2 = getMaskPointsInROIs(kps2_sift,ROIs2)
    kps2_filter, des2_filter = Img2.featureFilter(final_mask2)
    
    
    # BFMatches(des1_filter, des2_filter)
    matches_sift = utils.featureMatch(des1_filter, des2_filter, 'sift')
    matches_sift_knn = utils.featureMatch(des1_filter, des2_filter, 'sift', knn=True)
    #bf_matches = FLANNMatches(des1_filter,des2_filter)

    img_sift = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_sift[:50],None,**draw_params)
    img_sift_knn = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_sift_knn,None,**draw_params)
  
  
  
    kps1_brisk, dps1_brisk = Img1.findFeatures('brisk')
    kps2_brisk, dps2_brisk = Img2.findFeatures('brisk')
    
    final_mask1_brisk = getMaskPointsInROIs(kps1_brisk,ROIs1)
    kps1_filter_, des1_filter_ = Img1.featureFilter(final_mask1_brisk)

    final_mask2_brisk = getMaskPointsInROIs(kps2_brisk,ROIs2)
    kps2_filter_, des2_filter_ = Img2.featureFilter(final_mask2_brisk)
    
    
    matches_brisk = utils.featureMatch(des1_filter_, des2_filter_, 'brisk')
    matches_brisk_knn = utils.featureMatch(des1_filter_, des2_filter_, 'brisk',knn=True)



    img_brisk = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_brisk[:50],None,**draw_params)
    img_brisk_knn = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_brisk_knn,None,**draw_params)
    
    
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.title("Brute Force Matching on SIFT Features\n(# of Matches: %d)" %(len(matches_sift)))
    plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.title("Brute Force KNN Matching on SIFT Features\n(# of Matches: %d)" %(len(matches_sift_knn)))
    plt.imshow(cv.cvtColor(img_sift_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.title("Brute Force Matching on BRISK Features\n(# of Matches: %d)" %(len(matches_brisk)))
    plt.imshow(cv.cvtColor(img_brisk, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.title("Brute Force KNN Matching on BRISK Features\n(# of Matches: %d)" %(len(matches_brisk_knn)))
    plt.imshow(cv.cvtColor(img_brisk_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    

    homo_mat_sift, inliers_mask_sift = findHomography(matches_sift,kps1_filter,kps2_filter)
    homo_mat_sift_knn, inliers_mask_sift_knn = findHomography(matches_sift_knn,kps1_filter,kps2_filter)
    homo_mat_brisk, inliers_mask_brisk = findHomography(matches_brisk,kps1_filter_,kps2_filter_)
    homo_mat_brisk_knn, inliers_mask_brisk_knn = findHomography(matches_brisk_knn,kps1_filter_,kps2_filter_)
    
    
    matches_inliers_sift = list(itertools.compress(matches_sift, inliers_mask_sift))
    matches_inliers_sift_knn = list(itertools.compress(matches_sift_knn, inliers_mask_sift_knn))
    matches_inliers_brisk = list(itertools.compress(matches_brisk, inliers_mask_brisk))
    matches_inliers_brisk_knn = list(itertools.compress(matches_brisk_knn, inliers_mask_brisk_knn))
    
    
    img_inliners_sift = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers_sift,None,**draw_params)
    img_inliners_sift_knn = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers_sift_knn,None,**draw_params)
    img_inliners_brisk = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_inliers_brisk,None,**draw_params)
    img_inliners_brisk_knn = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_inliers_brisk_knn,None,**draw_params)    
    
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.title("Brute Force Matching on SIFT Features\n(# of Inliners: %d)" %(len(matches_inliers_sift)))
    plt.imshow(cv.cvtColor(img_inliners_sift, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.title("Brute Force KNN Matching on SIFT Features\n(# of Inliners: %d)" %(len(matches_inliers_sift_knn)))
    plt.imshow(cv.cvtColor(img_inliners_sift_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.title("Brute Force Matching on BRISK Features\n(# of Inliners: %d)" %(len(matches_inliers_brisk)))
    plt.imshow(cv.cvtColor(img_inliners_brisk, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.title("Brute Force KNN Matching on BRISK Features\n(# of Inliners: %d)" %(len(matches_inliers_brisk_knn)))
    plt.imshow(cv.cvtColor(img_inliners_brisk_knn, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

    plt.axis('off')