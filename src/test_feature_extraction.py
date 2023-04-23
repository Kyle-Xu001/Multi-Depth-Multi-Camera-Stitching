import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from stitch import utils, Image

if __name__ == '__main__':
    """
    This script is used for feature extraction testing.
    """
    # Define parser arguments
    parser = argparse.ArgumentParser(description="Image Stitching")
    parser.add_argument("--img", type=str)
    parser.add_argument("--rotate", action="store_true" , help="Rotate the image to get better visualization")
    args, _ = parser.parse_known_args()
    
    # Load exmaple image
    img = cv.imread(args.img)
    # img = cv.imread("dataset/example_image/APAP-railtracks/1.JPG")
    # img = cv.imread("dataset/Arie/lamp_01_Arie.PNG")
    # img = cv.imread("dataset/example_image/park/1.jpg")
    
    if bool(args.rotate):
        img = np.rot90(img,1) # Rotate the image to get better visualization
    
    Img = Image(img)
    
    # Original Image
    Img_copy = Image(img)
    
    # Equalized Image
    Img.equalizeHist()
    
    # Old_Equalized Image
    img_ = utils.equalizeHist_old(img)
    Img_ = Image(img_)

    # show the original image and equal-histogram image
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(Img_copy.img, cv.COLOR_BGR2RGB))
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
    
    fig.tight_layout()
    plt.show()
    
    # Extract the features from image
    kps_sift, dps_sift = Img.findFeatures('sift')
    kps_brisk, dps_brisk = Img.findFeatures('brisk')
    kps_orb, dps_orb = Img.findFeatures('orb')
    # show the original image and equalhist image
    
    fig1 = plt.figure(figsize=(15, 10))
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
    fig2 = plt.figure(figsize=(15, 10))
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
    
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
