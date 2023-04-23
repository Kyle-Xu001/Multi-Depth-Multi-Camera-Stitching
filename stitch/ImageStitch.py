import itertools
import numpy as np
import cv2 as cv
from scipy.interpolate import Rbf
from . import utils

'''
IMAGE Class: 

Define the feature properties for single image
----------
self.img : Camera Image from Single Frame [np.array]
self.nfeatures : Number of Total Features [int]
self.kps : Total Key Points (Features) [tuple]
self.des : Total Key Points Descriptors [np.array]
self.kpsCluster : Cluster List of Key Points(Features) [list of tuple]
self.desCluster : Cluster List of Descriptors [list of np.array]
'''
class Image(object):
    # Initialization for the Image object
    def __init__(self, img, nfeatures=0, method = 'sift'):
        self.img = img
        self.nfeatures = nfeatures
        self.kps = None # Position of Total Features 
        self.des = None
        self.kpsCluster = None
        self.desCluster = None
        
        # Equalize the YUV channels histogram
        self.equalizeHist()
        
        # Extract the features from image and update the feature property
        kps, des = self.findFeatures(method)
        print("\nInitial Successfully!")
        
    def equalizeHist(self):
        self.img = utils.equalizeHist(self.img)        

    # Finding all features on the Object Image
    def findFeatures(self, method=None):
        img = self.img
        kps, des = utils.findFeatures(img, method)
        
        self.kps = kps
        self.des = des
        self.nfeatures = len(kps)
        
        return kps, des
        
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
        homo_mat, inliers_mask = utils.findHomography(self.matches, self.Img1.kps, self.Img2.kps)
        matches_inliers = list(itertools.compress(self.matches, inliers_mask))
        
        return homo_mat, matches_inliers
    
    
    def clusterMatch(self, method, knn=False):
        matches_list = []
        for i in range(len(self.Img1.desCluster)):
            des1 = self.Img1.desCluster[i]
            des2 = self.Img2.desCluster[i]
            
            matches = utils.featureMatch(des1, des2, method, knn)
            matches_list.append(matches)
        return matches_list
    
    
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

def remapStitch(img1, img2, u, v, map_x, map_y):
    # x_range = np.arange(0, stitch_size[0]+1)
    # y_range = np.arange(0, stitch_size[1]+1)
    # u, v = np.meshgrid(x_range, y_range)
    # u = np.float32(u)
    # v = np.float32(v)
    img_super = cv.remap(img1, u, v, cv.INTER_LINEAR)
    img_transform = cv.remap(img2, map_x, map_y, cv.INTER_LINEAR)
    
    #img_transform[high_y:high_y + 200, :, :] = 0
    #img_super[mask] = 0
    img_transform[:600, :, :] = 0
    img_super[600:, :, :] = 0

    img_stitch = img_transform + img_super
    return img_stitch

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
    offset = np.minimum(np.abs(y_min), img2.shape[0])
    width = np.maximum(np.abs(x_max-x_min), x_max)
    height = np.maximum(np.abs(y_max-y_min), y_max)
    stitch_size = (width, height)
    translation = np.array([[0,0,0],
                            [0,0,offset],
                            [0,0,0]])
    homo_mat_ = np.eye(3)
    homo_mat  = homo_mat  + translation
    homo_mat_ = homo_mat_ + translation
    img_super = cv.warpPerspective(
        img1, homo_mat_, stitch_size)
    img_transform = cv.warpPerspective(
        img2, homo_mat, stitch_size)

    
    # Combine the image on one super image
    high_y = np.min(posVerts[:, 1])
    img_transform[high_y:high_y, :, :] = 0
    img_super[img_transform > 0] = 0

    img_stitch = img_transform + img_super

    return img_stitch

def alphablend(img1, img2, homo_mat,trim_len=0,alpha = 0.5):
    # Get the position of vertices
    posVerts = utils.transformVerts(img_size=np.array(
        [img2.shape[1], img2.shape[0]]), homo_mat=homo_mat)

    x_min = posVerts[:, 0].min()
    x_max = posVerts[:, 0].max()
    y_min = posVerts[:, 1].min()
    y_max = posVerts[:, 1].max()
    # print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %
    #       (x_min, x_max, y_min, y_max))

    # Define the size of the result image
    stitch_size = (x_max, y_max)

    homo_mat_ = np.eye(3)
    img_super = cv.warpPerspective(
        img1, homo_mat_, stitch_size, borderValue=(0, 0, 0))
    img_transform = cv.warpPerspective(
        img2, homo_mat, stitch_size, borderValue=(0, 0, 0))
    
    beta = (1.0 - alpha)
    
    high_y = np.min(posVerts[:,1])
    img_transform[high_y:high_y+trim_len,:,:] = (alpha *  img_transform[high_y:high_y+trim_len,:,:]).astype(np.uint8)
    
    img_transform[img_super>0] = ((beta) * img_transform[img_super>0]).astype(np.uint8)
    img_super[img_transform > 0] =  ((alpha) * img_super[img_transform > 0]).astype(np.uint8)
    super_img = cv.addWeighted(img_super, 1, img_transform, 1, 0.0)
    
    return super_img