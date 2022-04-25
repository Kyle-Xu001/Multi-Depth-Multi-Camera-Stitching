from matplotlib import pyplot as plt
import numpy as np
import undistortion
import argparse
import cv2 as cv

def order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts, dst):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    maxHeight = max(int(heightA),int(heightB))
    
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
    '''
    Print the parameters of homography matrix
    '''
    np.set_printoptions(suppress=True)
    print(M.flatten().tolist())
    
    return warped


if __name__ == '__main__':
    # Load the target image
    image = cv.imread("dataset/lamp_07_030715.PNG")
    
    # Undistorted the oringal image
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"
    lamp_id = 'lamp07'
    image, _, _ = undistortion.undistort(image, lamp_id, calib_dir)
    
    #image = cv.rotate(image,cv.ROTATE_90_COUNTERCLOCKWISE)
    
    
    plt.figure(1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    # Select the target four points for transformation
    pts = np.array([
        [0, 0],
        [1376, 0],
        [1000,550],
        [300, 550]])
    
    dst = np.array([
        [0, 0],
        [1376, 0],
        [1200, 450],
        [300, 450]],dtype="float32")
    
    warped = four_point_transform(image, pts, dst)
    
    # Show the transformed result
    plt.figure(2)
    plt.imshow(cv.cvtColor(warped, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()