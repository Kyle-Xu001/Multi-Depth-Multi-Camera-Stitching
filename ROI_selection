import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # Read Image
    img1 = cv.imread("dataset/origin_images/lamp_17_distorted_empty_for16.PNG")
    img2 = cv.imread("dataset/origin_images/lamp_16_distorted_empty_for17.PNG")
    plt.figure(0)
    plt.subplot(2,1,1)
    plt.imshow(img1)
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.imshow(img2)
    plt.axis('off')
    plt.show()
    # Select ROIs
    ROIs = cv.selectROIs("select the area", img1)
    
    # loop over every bounding box
    for index,rect in enumerate(ROIs):
        img_select = img1[int(rect[1]):int(rect[1]+rect[3]), 
                      int(rect[0]):int(rect[0]+rect[2])]
        
        cv.imshow('Cropped image',img_select)
        cv.waitKey(0)
        
        ROIs[index,2] = rect[0]+rect[2]
        ROIs[index,3] = rect[1]+rect[3]
        
    '''
    Print the parameters of ROI regions
    '''
    np.set_printoptions(suppress=True)
    print(ROIs.flatten().tolist())
    
    plt.show()