import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
import ImageStitch
from ImageStitch import Image,Stitch


# Define the main function
if __name__ == '__main__':
    # Define the draw parameters for matching visualization
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(0, 0, 255),
                       flags=cv.DrawMatchesFlags_DEFAULT)

    # load the initial images and corresponding homo matrix
    img1 = cv.imread("dataset/origin_images/lamp_15.JPG")
    img2 = cv.imread("dataset/origin_images/lamp_14.JPG")

    homo_mat = np.array([[1.010638759789845, -0.020425372912002478, -14.695282380876835, 0.0075814846672632484,
                        1.0066178703225623, 412.43074450438576, 2.1888462791375407e-05, -2.2383079441019405e-05, 1.0]]).reshape(-1, 3)
    
    '''
    Stitch the Images
    '''
    img_stitch = ImageStitch.simpleStitch(img1, img2, homo_mat)

    plt.figure(1)
    plt.imshow(cv.cvtColor(np.rot90(img_stitch), cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    

    # load the matching images
    img1 = cv.imread("dataset/origin_images/lamp_16.JPG")
    img2 = img_stitch

    ROIs1 = cv.selectROIs("select the area", img1)
    ROIs2 = cv.selectROIs("select the area", img2)

    for i in range(len(ROIs1)):
        ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
        ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
        ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
        ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]
        

    stitches = Stitch(img1, img2)
    stitches.featureExtract(ROIs1, ROIs2)
    homo_mat, matches_inliers = stitches.homoEstimate()
    kps1 = stitches.Img1.kps
    kps2 = stitches.Img2.kps
    
    
    # Visualization
    print("\nNumber of inlier matches: ", len(matches_inliers), "\n")
    img_inliers = cv.drawMatches(
            img1, kps1, img2, kps2, matches_inliers, None, **draw_params)

    plt.figure(2)
    plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
    plt.title("Inlier Matches for Total Selected Area")
    plt.axis('off')


    '''
    Stitch the Images
    '''
    img_stitch = ImageStitch.simpleStitch(img1, img2, homo_mat)

    plt.figure(3)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
