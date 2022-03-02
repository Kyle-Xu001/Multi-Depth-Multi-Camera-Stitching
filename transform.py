from matplotlib import pyplot as plt
import numpy as np
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
    print(image.shape[1])
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
    return warped


if __name__ == '__main__':
    image = cv.imread("lamp_19.JPG")
    image2 = cv.imread("lamp_18.JPG")
    
    plt.figure(1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis("off")

    
    plt.figure(2)
    plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    plt.axis("off")
    
    pts = np.array([
        [257, 0],
        [509, 0],
        [509, 767],
        [257, 767]
    ])
    
    dst = np.array([
        [205, 0],
        [400, 0],
        [400, 767],
        [205, 767]],dtype="float32")
    
    warped = four_point_transform(image, pts, dst)
    
    plt.imshow(cv.cvtColor(warped, cv.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.figure(4)
    plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", help = "path to the image file")
    # ap.add_argument("-c", "--coords",
    #                 help = "comma seperated list of source points")
    # args = vars(ap.parse_args())
    
    # image = cv.imread(args["image"])
    # pts = np.array(eval(args["coords"]),dtype="float32")
    
    # warped = four_point_transform(image, pts)
    
    # cv.imshow("Original", image)
    # cv.imshow("Warped", warped)
    # cv.waitKey(0)