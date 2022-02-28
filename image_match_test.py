import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import itertools

import image_struct_test as test

def BFMatches(des1, des2):
    bf = cv.BFMatcher_create()

    
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def BFMatches_orb(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def BFMatches_knn(des1, des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    matches_good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            matches_good.append(m)
    return matches_good


def FLANNMatches(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    #  Need to draw only good matches, so create a mask
    good = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            good.append(m)
    return good


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
    # load the matching images
    img1 = cv.imread("lamp_19_distorted.JPG")
    img2 = cv.imread("lamp_18_distorted.JPG")

    #img1 = img1[:,550:,:]
    #img2 = img2[:,550:,:]
    # Equalize histogram of images
    img1_hist = test.equalizeHist(img1)
    img2_hist = test.equalizeHist(img2)

    # Find SIFT features of matching images
    kps1_hist, dps1_hist = test.findFeatures(img1_hist)
    kps2_hist, dps2_hist = test.findFeatures(img2_hist)

    # Manual Set
    ROIs2 = np.array([[160,0,1250,400]])
    ROIs1 = np.array([[160,350,1250,760]])
    pts1_hist = cv.KeyPoint_convert(kps1_hist)
    final_mask1 = test.getMaskPointsInROIs(pts1_hist,ROIs1)
    kps1_filter, des1_filter = test.SIFT_filter(kps1_hist,dps1_hist,final_mask1)

    pts2_hist = cv.KeyPoint_convert(kps2_hist)
    final_mask2 = test.getMaskPointsInROIs(pts2_hist,ROIs2)
    kps2_filter, des2_filter = test.SIFT_filter(kps2_hist,dps2_hist,final_mask2)

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   flags = cv.DrawMatchesFlags_DEFAULT)
    
    # BFMatches(des1_filter, des2_filter)

    bf_matches = BFMatches_knn(des1_filter, des2_filter)
    #bf_matches = FLANNMatches(des1_filter,des2_filter)
    print('The number of matches: ', len(bf_matches))

    img3 = cv.drawMatches(img1_hist,kps1_filter,img2_hist,kps2_filter,bf_matches,None,**draw_params)
    
    
    
    kps1_hist_orb, dps1_hist_orb = test.findFeatures_orb(img1_hist)
    kps2_hist_orb, dps2_hist_orb = test.findFeatures_orb(img2_hist)

    pts1_hist_orb = cv.KeyPoint_convert(kps1_hist_orb)
    final_mask1_orb = test.getMaskPointsInROIs(pts1_hist_orb,ROIs1)
    kps1_filter_orb, des1_filter_orb = test.SIFT_filter(kps1_hist_orb,dps1_hist_orb,final_mask1_orb)

    pts2_hist_orb = cv.KeyPoint_convert(kps2_hist_orb)
    final_mask2_orb = test.getMaskPointsInROIs(pts2_hist_orb,ROIs2)
    kps2_filter_orb, des2_filter_orb = test.SIFT_filter(kps2_hist_orb,dps2_hist_orb,final_mask2_orb)

    bf_matches_orb = BFMatches_orb(des1_filter_orb, des2_filter_orb)
    print('The number of matches: ', len(bf_matches_orb))

    img3_orb = cv.drawMatches(img1_hist,kps1_filter_orb,img2_hist,kps2_filter_orb,bf_matches_orb[:10],None,**draw_params)
    
    plt.figure(1)
    #plt.subplot(1,2,1)
    plt.title("Brute Force Knn Matching on SIFT Features")
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.axis('off')

    # plt.subplot(1,3,3)
    # plt.title("Brute-Force Matching on ORB Features")
    # plt.imshow(cv.cvtColor(img3_orb, cv.COLOR_BGR2RGB))
    # plt.axis('off')

    homo_mat, inliers_mask = findHomography(bf_matches,kps1_filter,kps2_filter)
    np.set_printoptions(suppress=True)
    print(homo_mat.astype(float))
    matches_inliers = list(itertools.compress(bf_matches, inliers_mask))
    img_inliners = cv.drawMatches(img1_hist,kps1_filter,img2_hist,kps2_filter,matches_inliers,None,**draw_params)
    #plt.subplot(1,2,2)
    plt.title("Brute-Force Knn Matching on SIFT Features")
    plt.imshow(cv.cvtColor(img_inliners, cv.COLOR_BGR2RGB))
    plt.axis('off')


    plt.figure(2)


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


