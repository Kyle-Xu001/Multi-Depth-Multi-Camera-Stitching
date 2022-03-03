import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import utils


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
        
        # self.candidate_links = dict()
        # self.candidate_links['top'] = list()
        # self.candidate_links['bottom'] = list()
        # self.link = dict()
        print("\nInitial Successfully!")
        
    def equalizeHist(self):
        self.img = utils.equalizeHist(self.img)        

    def findFeatures(self, method=None):
        img = self.img
        kps, des = utils.findFeatures(img, method)
        
        self.kps = kps
        self.des = des
        self.nfeatures = len(kps)

        return kps, des
    
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
            
        return kpsCluster, desCluster
        

    
if __name__ == '__main__':
    # Load exmaple image
    img = cv.imread("lamp_19.JPG")
    img = np.rot90(img,1) # Rotate the image to get better visualization
    
    Img = Image(img)
    Img_copy = Image(img)
    
    img_ = utils.equalizeHist_old(Img.img)
    Img_ = Image(img_)



    # show the original image and equalhist image
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(Img.img, cv.COLOR_BGR2RGB))
    plt.title("(a) Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(Img_.img, cv.COLOR_BGR2RGB))
    plt.title("(b) RGB Equalized Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    Img.equalizeHist()
    plt.imshow(cv.cvtColor(Img.img, cv.COLOR_BGR2RGB))
    plt.title("(c) YUV Equalized Image")
    plt.axis('off')
    
    
    
    # Extract the features from image
    kps_sift, dps_sift = Img.findFeatures('sift')
    kps_brisk, dps_brisk = Img.findFeatures('brisk')
    kps_orb, dps_orb = Img.findFeatures('orb')
    # show the original image and equalhist image
    plt.figure(2)
    plt.subplot(1, 3, 1)
    img_kps_sift = cv.drawKeypoints(Img.img, kps_sift, None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_sift, cv.COLOR_BGR2RGB))
    plt.title("(a) SIFT Features")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    img_kps_brisk = cv.drawKeypoints(Img.img, kps_brisk, None, (0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_brisk, cv.COLOR_BGR2RGB))
    plt.title("(b) BRISK Features")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    img_kps_orb = cv.drawKeypoints(Img.img, kps_orb, None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_orb, cv.COLOR_BGR2RGB))
    plt.title("(c) ORB Features")
    plt.axis('off')



    # Compare the different equlization methods for feature extraction
    plt.figure(3)
    kps_sift_origin, dps_sift_origin = Img_copy.findFeatures('sift')
    kps_sift_old, dps_sift_old = Img_.findFeatures('sift')
    
    num_sift_origin = len(kps_sift_origin)
    num_sift_yuv = len(kps_sift)
    num_sift_rgb = len(kps_sift_old)
    
    kps_brisk_origin, dps_brisk_origin = Img_copy.findFeatures('brisk')
    kps_brisk_old, dps_brisk_old = Img_.findFeatures('brisk')
    
    num_brisk_origin = len(kps_brisk_origin)
    num_brisk_yuv = len(kps_brisk)
    num_brisk_rgb = len(kps_brisk_old)
        
    plt.subplot(2, 3, 1)
    img_kps_sift_origin = cv.drawKeypoints(Img_copy.img, kps_sift_origin, None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_sift_origin, cv.COLOR_BGR2RGB))
    plt.title("(a) SIFT Features for Original Image (#: %d)" %(num_sift_origin))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    img_kps_sift_old = cv.drawKeypoints(Img_.img, kps_sift_old, None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_sift_old, cv.COLOR_BGR2RGB))
    plt.title("(b) SIFT Features for RGB Equalize (#: %d)" %(num_sift_rgb))
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cv.cvtColor(img_kps_sift, cv.COLOR_BGR2RGB))
    plt.title("(c) SIFT Features for YUV Equalize (#: %d)" %(num_sift_yuv))
    plt.axis('off')
        
    plt.subplot(2, 3, 4)
    img_kps_brisk_origin = cv.drawKeypoints(Img_copy.img, kps_brisk_origin, None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_brisk_origin, cv.COLOR_BGR2RGB))
    plt.title("(d) BRISK Features for Original Image (#: %d)" %(num_brisk_origin))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    img_kps_brisk_old = cv.drawKeypoints(Img_.img, kps_brisk_old, None,(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    plt.imshow(cv.cvtColor(img_kps_brisk_old, cv.COLOR_BGR2RGB))
    plt.title("(e) BRISK Features for RGB Equalize (#: %d)" %(num_brisk_rgb))
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(cv.cvtColor(img_kps_brisk, cv.COLOR_BGR2RGB))
    plt.title("(f) BRISK Features for YUV Equalize (#: %d)" %(num_brisk_yuv))
    plt.axis('off')
    plt.show()