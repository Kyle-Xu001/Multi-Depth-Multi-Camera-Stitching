import itertools
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

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


def getMaskPointsInROIs(kps, ROIs):
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
        x_mask = np.logical_and(pts[:, 0] >= ROI[0], pts[:, 0] <= ROI[2])
        y_mask = np.logical_and(pts[:, 1] >= ROI[1], pts[:, 1] <= ROI[3])
        submasks.append(np.logical_and(x_mask, y_mask))

    # final_mask = np.zeros(submasks[0].shape,dtype=bool)
    # for mask in submasks:
    #     final_mask = np.logical_or(final_mask,mask)

    return submasks


def findFeatures(img, method=None):

    if method == 'sift':
        descriptor = cv.SIFT_create(contrastThreshold=0.01)
    elif method == 'brisk':
        descriptor = cv.BRISK_create()
    elif method == 'orb':
        descriptor = cv.ORB_create(nfeatures=5000, nlevels=8)

    kps, des = descriptor.detectAndCompute(img, None)

    return kps, des


def featureMatch(des1, des2, method, knn=False):
    if method == 'sift' and knn == False:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    elif method == 'sift' and knn == True:
        bf = cv.BFMatcher()
    elif method == 'brisk' and knn == False:
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    elif method == 'brisk' and knn == True:
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
        des2 = desCluster2[i]

        bf = cv.BFMatcher()

        match = bf.knnMatch(des1, des2, k=3)

        matchFilter = []
        for m, _, n in match:
            if m.distance < 0.9 * n.distance:
                matchFilter.append(m)
        matches.append(matchFilter)
    return matches


def findHomography(matches, kps1, kps2):
    queryIdxs = [match.queryIdx for match in matches]
    trainIdxs = [match.trainIdx for match in matches]
    kps2 = cv.KeyPoint_convert(kps2)
    kps1 = cv.KeyPoint_convert(kps1)
    homo_mat, inliers_mask = cv.findHomography(
        kps2[trainIdxs], kps1[queryIdxs], method=cv.RANSAC, ransacReprojThreshold=30)

    return homo_mat, inliers_mask


def matchFilter(matches, invalid_index1, invalid_index2):
    new_matches = []
    for match in matches:
        if (match.queryIdx not in invalid_index1) and (match.trainIdx not in invalid_index2):
            new_matches.append(match)

    return new_matches


def drawMatch(Img1, kpsCluster1, Img2, kpsCluster2, matches, params):
    numCluster = len(kpsCluster1)

    # Get the num of columns to show the inmages
    if numCluster % 2 == 0:
        imageNum = int(numCluster/2)
        for i in range(numCluster):
            plt.figure(0)
            plt.subplot(2, imageNum, i+1)
            plt.title('Feature Matching in Area %d' % (i+1))
            img_match = cv.drawMatches(
                Img1.img, kpsCluster1[i], Img2.img, kpsCluster2[i], matches[i], None, **params)
            plt.axis('off')
            plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    else:
        imageNum = (numCluster+1)/2
        for i in range(numCluster):
            plt.figure(0)
            plt.subplot(imageNum, 2, i+1)
            plt.title('Feature Matching in Area %d' % (i+1))
            img_match = cv.drawMatches(
                Img1.img, kpsCluster1[i], Img2.img, kpsCluster2[i], matches[i], None, **params)
            plt.axis('off')
            plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))

# Check if the matching inliers are correct


def inlierChecker(Img1, kps1, Img2, kps2, inliers, params, num):
    for i in range(len(inliers)):
        plt.figure(4)
        img = cv.drawMatches(Img1.img, kps1, Img2.img, kps2,
                             inliers[num:num+1], None, **params)
        plt.title("Checking Matching Inlier [%d]" % (num))
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.axis('off')


'''
Generate the feature density map
'''


def featureHeatmap(image, features):
    # Transfer the original image into background
    grayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.normalize(grayImg, grayImg, 0, 125, cv.NORM_MINMAX)
    normImg = np.asarray(grayImg, dtype=np.uint8)

    # Define the Guassian Distribution Matrix
    dsize = 21
    offset = int((dsize - 1)/2)
    guassian_pitch = guassian(dsize=dsize, sigma=5)
    featImg = np.zeros(grayImg.shape)

    # Add the feature at the feature map
    features = cv.KeyPoint_convert(features)
    for feature in features:
        if (feature[0] > 11 and feature[0] < image.shape[1]-11) and (feature[1] > 11 and feature[1] < image.shape[0]-11):
            featImg[int(feature[1])-offset:int(feature[1])+offset+1, int(feature[0]
                                                                         )-offset:int(feature[0])+offset+1] += 3500*guassian_pitch
    featImg = np.asarray(featImg, dtype=np.uint8)

    heatImg = cv.applyColorMap(featImg, cv.COLORMAP_JET)

    normImg = cv.applyColorMap(normImg, cv.COLORMAP_JET)

    # Combine the original image to the density distribution image
    heatImg = cv.addWeighted(normImg, 0.5, heatImg, 0.9, 0)
    heatImg = cv.cvtColor(heatImg, cv.COLOR_BGR2RGB)

    return heatImg


'''
Generate Guassian Distribution in 2D Matrix
'''


def guassian(dsize, sigma):
    # Declare the guassian matrix
    guassian_pitch = np.zeros([dsize, dsize])
    offset = (dsize-1)/2

    for i in range(dsize):
        for j in range(dsize):
            guassian_pitch[i, j] = math.exp(
                (-1/(2*sigma**2))*(np.square(i-offset) + np.square(j-offset)))/(2*math.pi*sigma**2)

    # Normailize the Guassian Matrix
    guassian_sum = np.sum(guassian_pitch)
    guassian_pitch = guassian_pitch/guassian_sum

    return guassian_pitch


def transformVerts(img_size, homo_mat):
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
    x1 = np.array([0, 0])
    x2 = np.array([img_size[0], 0])
    x3 = np.array([img_size[0], img_size[1]])
    x4 = np.array([0, img_size[1]])

    X = np.zeros([1, 4, 2])
    X[:, 0, :] = x1
    X[:, 1, :] = x2
    X[:, 2, :] = x3
    X[:, 3, :] = x4
    X_transform = cv.perspectiveTransform(X, homo_mat)

    return X_transform.round().astype(np.int32).reshape(4, 2)
