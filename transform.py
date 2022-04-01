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
    
    M = cv.getPerspectiveTransform(rect, dst)
    print(M)
    warped = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped


if __name__ == '__main__':
    image = cv.imread("stitch.PNG")
    #image2 = cv.imread("lamp_18.JPG")
    
    plt.figure(1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis("off")

    
    #plt.figure(2)
    #plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    #plt.axis("off")
    
    pts = np.array([
        [200, 3600],
        [1150,2850],
        [1250, 4250],
        [0, 5450]
    ])
    
    dst = np.array([
        [250, 3300],
        [1150, 2850],
        [1350, 4250],
        [0, 4500]],dtype="float32")
    
    warped = four_point_transform(image, pts, dst)
    plt.figure(2)
    plt.imshow(cv.cvtColor(np.rot90(warped), cv.COLOR_BGR2RGB))
    plt.axis("off")
    
    # plt.figure(4)
    # plt.imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    # plt.axis("off")
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