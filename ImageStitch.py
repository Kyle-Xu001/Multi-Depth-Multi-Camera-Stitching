import itertools
import numpy as np
import cv2 as cv

import utils

'''
IMAGE Class: 
Define the feature properties for single image
----------
'''
class Image(object):
    # Initialization for the Image object
    def __init__(self, img, nfeatures=0):
        self.img = img
        self.nfeatures = nfeatures
        self.kps = None
        self.des = None
        self.kpsCluster = None
        self.desCluster = None
        
        # Equalize the YUV channels histogram
        self.equalizeHist()
        
        # Extract the features from image and update the feature property
        self.findFeatures('sift')
        print("\nInitial Successfullys")
        
    def equalizeHist(self):
        self.img = utils.equalizeHist(self.img)        

    # Finding all features on the Object Image
    def findFeatures(self, method=None):
        img = self.img
        kps, des = utils.findFeatures(img, method)
        
        self.kps = kps
        self.des = des
        self.nfeatures = len(kps)
    
    # Select features in the target region (single region)
    def featureFilter(self, mask):
        kps_filtered = ()
        des_filtered = []
        
        # Filter the key points and descriptions within the range
        for i in range(len(mask)):
            if mask[i] == 1:
                kps_filtered = kps_filtered + (self.kps[i],)
                des_filtered.append(self.des[i])
    
        des_filtered = np.asarray(des_filtered)
        
        print("\nTotal features: ", len(mask))
        print("Filtered features: ", len(kps_filtered),"\n")
        
        return kps_filtered, des_filtered
    
    # Divide all features into several clusters
    def featureCluster(self, masks):
        kpsCluster = []
        desCluster = []
        
        # Filter the features in the same range in one cluster
        for mask in masks:
            kps_filtered, des_filtered = self.featureFilter(mask)
            kpsCluster.append(kps_filtered)
            desCluster.append(des_filtered)
        
        self.kpsCluster = kpsCluster
        self.desCluster = desCluster

 
'''
STITCH CLASS: 
Define the related features and matches between a pair of images 
----------
`featureExtract`: extract features within each image and classify them in given ROIs
'''
class Stitch(object):
    def __init__(self, img1, img2):
        self.Img1 = Image(img1)
        self.Img2 = Image(img2)
        self.matches = None
        
    def featureExtract(self, ROIs1, ROIs2):
        masks1 = utils.getMaskPointsInROIs(self.Img1.kps, ROIs1)
        masks2 = utils.getMaskPointsInROIs(self.Img2.kps, ROIs2)
        
        self.Img1.featureCluster(masks1)
        self.Img2.featureCluster(masks2)
    
    def homoEstimate(self):
        # Define the matches based on two images
        matches_list = utils.clusterMatch(self.Img1.desCluster, self.Img2.desCluster)

        # Combine the features in one lists from each cluster
        self.featureIntegrate(matches_list)
        
        homo_mat, inliers_mask = utils.findHomography(self.matches, self.Img1.kps, self.Img2.kps)
        matches_inliers = list(itertools.compress(self.matches, inliers_mask))
        
        return homo_mat, matches_inliers
    
    def featureIntegrate(self, matches_list):
        # Define the variables
        kpsCluster1 = self.Img1.kpsCluster
        kpsCluster2 = self.Img2.kpsCluster
        numCluster = len(kpsCluster1)
        assert(numCluster == len(kpsCluster2))

        kps1 = ()
        kps2 = ()
        matchInt = []
        queryIdx = 0
        trainIdx = 0

        # Integrete the features from each cluster into one list
        for i in range(numCluster):
            kps1 = kps1 + kpsCluster1[i]
            kps2 = kps2 + kpsCluster2[i]

            # Update the index for each match cluster
            for j in range(len(matches_list[i])):
                matches_list[i][j].queryIdx = matches_list[i][j].queryIdx + queryIdx
                matches_list[i][j].trainIdx = matches_list[i][j].trainIdx + trainIdx

            queryIdx = queryIdx + len(kpsCluster1[i])
            trainIdx = trainIdx + len(kpsCluster2[i])

            # Combine the matches into one list
            matchInt = matchInt + matches_list[i]
        
        # Update the features and matches in class
        self.Img1.kps = kps1
        self.Img2.kps = kps2
        self.matches = matchInt
    

def simpleStitch(img1, img2, homo_mat):
    # Get the position of vertices
    posVerts = utils.transformVerts(img_size=np.array(
        [img2.shape[1], img2.shape[0]]), homo_mat=homo_mat)

    x_min = posVerts[:, 0].min()
    x_max = posVerts[:, 0].max()
    y_min = posVerts[:, 1].min()
    y_max = posVerts[:, 1].max()
    print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %
          (x_min, x_max, y_min, y_max))

    # Define the size of the result image
    stitch_size = (x_max, y_max)

    homo_mat_ = np.eye(3)
    img_super = cv.warpPerspective(
        img1, homo_mat_, stitch_size, borderValue=(0, 0, 0))
    img_transform = cv.warpPerspective(
        img2, homo_mat, stitch_size, borderValue=(0, 0, 0))

    # Combine the image on one super image
    high_y = np.min(posVerts[:, 1])
    img_transform[high_y:high_y, :, :] = 0
    img_super[img_transform > 0] = 0

    img_stitch = img_transform + img_super

    return img_stitch