import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import itertools

from image_feature_extraction_test import Image
import utils

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
            if m.distance < 0.75*n.distance:
                matches_good.append(m)
                
    return matches_good


def findHomography(matches, kps1, kps2):
    queryIdxs = [match.queryIdx for match in matches]
    trainIdxs = [match.trainIdx for match in matches]
    kps2 = cv.KeyPoint_convert(kps2)
    kps1 = cv.KeyPoint_convert(kps1)
    homo_mat,inliers_mask = cv.findHomography(kps2[trainIdxs],kps1[queryIdxs],method=cv.RANSAC,ransacReprojThreshold=2)

    return homo_mat, inliers_mask

def transformVerts(img_size,homo_mat):
    """
    Finds the vertices of image of img_size transformed by homo_mat

    Parameters
    ----------
    img_size : Size 2 int iterable
        (width,height)
    homo_mat : 3x3 numpy array
        Homography matrix

    Returns
    -------
    4x2 numpy array
        Array where each row is a vertice, first column is the x coordinate,
        second column is the y coordinate.

    """
    x1 = np.array([0,0])
    x2 = np.array([img_size[0],0])
    x3 = np.array([img_size[0],img_size[1]])
    x4 = np.array([0,img_size[1]])
    
    X = np.zeros([1,4,2])
    X[:,0,:] = x1
    X[:,1,:] = x2
    X[:,2,:] = x3
    X[:,3,:] = x4
    X_transform = cv.perspectiveTransform(X,homo_mat)
    
    return X_transform.round().astype(np.int32).reshape(4,2)

if __name__ == '__main__':
    
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    
    # load the matching images
    img1 = cv.imread("lamp_16.JPG")
    img2 = cv.imread("lamp_15.JPG")

    img1 = np.rot90(img1,1) 
    img2 = np.rot90(img2,1)
    
    img1 = utils.equalizeHist_old(img1)
    img2 = utils.equalizeHist_old(img2)
    
    # Equalize histogram of images
    Img1 = Image(img1)
    Img2 = Image(img2)
    
    # Img1.equalizeHist()
    # Img2.equalizeHist()



    # Find SIFT features of matching images
    kps1_sift, dps1_sift = Img1.findFeatures('sift')
    kps2_sift, dps2_sift = Img2.findFeatures('sift')

    # Manual Set
    # ROIs1 = np.array([[160,350,1250,760]])
    # ROIs2 = np.array([[160,0,1250,400]])
    ROIs1 = np.array([[350,150,760,1250]])
    ROIs2 = np.array([[0,150,400,1250]])
    
    pts1_sift = cv.KeyPoint_convert(kps1_sift)
    final_mask1 = utils.getMaskPointsInROIs(pts1_sift,ROIs1)
    kps1_filter, des1_filter = Img1.featureFilter(final_mask1)

    pts2_sift = cv.KeyPoint_convert(kps2_sift)
    final_mask2 = utils.getMaskPointsInROIs(pts2_sift,ROIs2)
    kps2_filter, des2_filter = Img2.featureFilter(final_mask2)
    
    
    # BFMatches(des1_filter, des2_filter)
    matches_sift = featureMatch(des1_filter, des2_filter, 'sift')
    matches_sift_knn = featureMatch(des1_filter, des2_filter, 'sift', knn=True)
    #bf_matches = FLANNMatches(des1_filter,des2_filter)

    img_sift = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_sift[:50],None,**draw_params)
    img_sift_knn = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_sift_knn,None,**draw_params)
  
  
  
    kps1_brisk, dps1_brisk = Img1.findFeatures('brisk')
    kps2_brisk, dps2_brisk = Img2.findFeatures('brisk')
    
    pts1_brisk = cv.KeyPoint_convert(kps1_brisk)
    final_mask1_brisk = utils.getMaskPointsInROIs(pts1_brisk,ROIs1)
    kps1_filter_, des1_filter_ = Img1.featureFilter(final_mask1_brisk)

    pts2_brisk = cv.KeyPoint_convert(kps2_brisk)
    final_mask2_brisk = utils.getMaskPointsInROIs(pts2_brisk,ROIs2)
    kps2_filter_, des2_filter_ = Img2.featureFilter(final_mask2_brisk)
    
    
    matches_brisk = featureMatch(des1_filter_, des2_filter_, 'brisk')
    matches_brisk_knn = featureMatch(des1_filter_, des2_filter_, 'brisk',knn=True)
    #bf_matches = FLANNMatches(des1_filter,des2_filter)
    print('The number of BRISK matches: ', len(matches_brisk))
    print('The number of BRISK Knn matches: ', len(matches_brisk_knn))

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
    homo_mat_brisk_knn, inliers_mask_brisk_knn = findHomography(matches_sift,kps1_filter_,kps2_filter_)
    
    
    matches_inliers_sift = list(itertools.compress(matches_sift, inliers_mask))
    matches_inliers = list(itertools.compress(matches_sift, inliers_mask))
    matches_inliers = list(itertools.compress(matches_sift, inliers_mask))
    matches_inliers = list(itertools.compress(matches_sift, inliers_mask))
    
    
    img_inliners = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers,None,**draw_params)
    plt.figure(2)
    # plt.subplot(1,2,2)
    plt.title("Brute-Force Knn Matching on SIFT Features")
    plt.imshow(cv.cvtColor(img_inliners, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()





    #img_transform = cv.warpPerspective(img1_hist, homo_mat,(img2_hist.shape[1],int(max(left_bottom[0],right_bottom[0]))))

   
    #img_transform[453:,:,:]=img2_hist

   

    plt.axis('off')

    #plt.show()


    # Get the position of vertices
    posVerts = transformVerts(img_size=np.array([img1_hist.shape[1],img1_hist.shape[0]]), homo_mat=homo_mat)
    print("Left Top: ",posVerts[0,:],"\n"
          "Right Top: ",posVerts[1,:],"\n"
          "Right Bottom: ",posVerts[2,:],"\n"
          "Left Bottom: ",posVerts[3,:],"\n")
    
    x_min = posVerts[:,0].min()
    x_max = posVerts[:,0].max()
    y_min = posVerts[:,1].min()
    y_max = posVerts[:,1].max()
    print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %(x_min,x_max,y_min,y_max))

    stitch_size = (x_max,y_max)

    homo_mat_ = np.eye(3)
    img_super = cv.warpPerspective(img1_hist, homo_mat_,stitch_size,borderValue=(0,0,0))

    img_transform = cv.warpPerspective(img2_hist, homo_mat,stitch_size,borderValue=(0,0,0))

    high_y = np.min(posVerts[:,1])
    img_transform[high_y:high_y,:,:] = 0

    img_super[img_transform>0]=0
    img_super = img_transform + img_super

    img_stitch = cv.rotate(img_super,cv.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.show()


