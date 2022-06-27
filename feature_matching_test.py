import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import itertools

from stitch import Image, utils


if __name__ == '__main__':
    '''This script will be tested for feature matching'''
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    
    # load the matching images
    #img1 = cv.imread("dataset/Arie/lamp_02_Arie.PNG")
    #img2 = cv.imread("dataset/Arie/lamp_01_Arie.PNG")
    img1 = cv.imread("dataset/example_image/APAP-railtracks/1.JPG")
    img2 = cv.imread("dataset/example_image/APAP-railtracks/2.JPG")
    #img1 = cv.imread("dataset/example_image/NISwGSP-denny/denny02.jpg")
    #img2 = cv.imread("dataset/example_image/NISwGSP-denny/denny03.jpg")

    #img1 = np.rot90(img1,1) 
    #img2 = np.rot90(img2,1)
    
    # Initialize the Stitch Class
    Img1 = Image(img1)
    Img2 = Image(img2)

    # Manually define the ROI to locate the area for corresponding images
    cv.namedWindow("Area Selection", cv.WINDOW_NORMAL)
    cv.resizeWindow("Area Selection", 800, 600)
    ROIs1 = cv.selectROIs("Area Selection", img1)
    ROIs2 = cv.selectROIs("Area Selection", img2)

    for i in range(len(ROIs1)):
        ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
        ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
        ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
        ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]
        
        
    '''SIFT Features Matching Comparison'''
    # Find SIFT features of matching images
    kps1_sift, dps1_sift = Img1.kps, Img1.des
    kps2_sift, dps2_sift = Img2.kps, Img2.des
    
    # Filter the Features
    masks1 = utils.getMaskPointsInROIs(kps1_sift, ROIs1)
    masks2 = utils.getMaskPointsInROIs(kps2_sift, ROIs2)
    
    kps1_filter, des1_filter = Img1.featureFilter(masks1[0])
    kps2_filter, des2_filter = Img2.featureFilter(masks2[0])
    

    # BFMatches(des1_filter, des2_filter)
    matches_sift = utils.featureMatch(des1_filter, des2_filter, 'sift')
    matches_sift_knn = utils.featureMatch(des1_filter, des2_filter, 'sift', knn=True)
    

    img_sift = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_sift[:300],None,**draw_params)
    img_sift_knn = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_sift_knn[:300],None,**draw_params)
  
    
    
    '''BRISK Features Matching Comparison'''
    # Extract the BRISK features
    kps1_brisk, dps1_brisk = Img1.findFeatures('brisk')
    kps2_brisk, dps2_brisk = Img2.findFeatures('brisk')
    
    # Filter the Features
    mask1_brisk = utils.getMaskPointsInROIs(kps1_brisk,ROIs1)
    kps1_filter_, des1_filter_ = Img1.featureFilter(mask1_brisk[0])

    mask2_brisk = utils.getMaskPointsInROIs(kps2_brisk,ROIs2)
    kps2_filter_, des2_filter_ = Img2.featureFilter(mask2_brisk[0])
    
    # BFMatches(des1_filter, des2_filter)
    matches_brisk = utils.featureMatch(des1_filter_, des2_filter_, 'brisk')
    matches_brisk_knn = utils.featureMatch(des1_filter_, des2_filter_, 'brisk',knn=True)

    img_brisk = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_brisk[:300],None,**draw_params)
    img_brisk_knn = cv.drawMatches(Img1.img,kps1_filter_,Img2.img,kps2_filter_,matches_brisk_knn,None,**draw_params)
    
    
    '''Visualize the Feature Matching Result'''
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
    plt.show()
    

    '''Estimate the Inliers'''
    homo_mat_sift, inliers_mask_sift = utils.findHomography(matches_sift,kps1_filter,kps2_filter)
    homo_mat_sift_knn, inliers_mask_sift_knn = utils.findHomography(matches_sift_knn,kps1_filter,kps2_filter)
    homo_mat_brisk, inliers_mask_brisk = utils.findHomography(matches_brisk,kps1_filter_,kps2_filter_)
    homo_mat_brisk_knn, inliers_mask_brisk_knn = utils.findHomography(matches_brisk_knn,kps1_filter_,kps2_filter_)
    
    matches_inliers_sift = list(itertools.compress(matches_sift, inliers_mask_sift))
    matches_inliers_sift_knn = list(itertools.compress(matches_sift_knn, inliers_mask_sift_knn))
    matches_inliers_brisk = list(itertools.compress(matches_brisk, inliers_mask_brisk))
    matches_inliers_brisk_knn = list(itertools.compress(matches_brisk_knn, inliers_mask_brisk_knn))
    
    '''Draw the Inliers'''
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