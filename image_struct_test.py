import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Image(object):
    def __init__(self, img, img_id=None, nfeatures=0):
        self.img = img
        self.img_id = img_id
        self.nfeatures = nfeatures

        self.kps = None
        self.des = None
        self.kps_histequal = None
        self.des_histequal = None

        self.candidate_links = dict()
        self.candidate_links['top'] = list()
        self.candidate_links['bottom'] = list()
        self.link = dict()
        print("\nInitial Successfully!")

    def findFeatures(self, hist_equal=True):
        if hist_equal and self.kps_histequal is None:
            img = equalizeHist(self.img)
        elif not hist_equal and self.kps is None:
            img = self.img
        else:  # The requested features are already computed
            return True

        kps, des = findFeatures(img)
        if len(kps) == 0:
            return False
        kps = cv.KeyPoint_convert(kps)

        if hist_equal:
            self.kps_histequal = kps
            self.des_histequal = des
        else:
            self.kps = kps
            self.des = des

        self.nfeatures = len(kps)
        return True


def findFeatures(img):

    #if method == 'sift':
    descriptor = cv.SIFT_create()
    #elif method == 'brisk':
        #descriptor = cv.BRISK_create()
    #elif method == 'orb':
        #descriptor = cv.ORB_create()
    
    kps, des = descriptor.detectAndCompute(img, None)

    # orb = cv.ORB_create(nfeatures=3000,nlevels=8)
    # kps,des = orb.detectAndCompute(img,None)
    # print(kps)

    return kps, des

# def findFeatures_orb(img):
#     # sift = cv.SIFT_create()
#     # kps, des = sift.detectAndCompute(img, None)

#     orb = cv.ORB_create(nfeatures=3000,nlevels=8)
#     kps,des = orb.detectAndCompute(img,None)

#     return kps, des


def equalizeHist(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    img_histequal = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    # img_histequal_ = np.zeros(img.shape, dtype=np.uint8)
    # img_histequal_[:, :, 0] = cv.equalizeHist(img[:, :, 0])
    # img_histequal_[:, :, 1] = cv.equalizeHist(img[:, :, 1])
    # img_histequal_[:, :, 2] = cv.equalizeHist(img[:, :, 2])

    return img_histequal

def getMaskPointsInROIs(pts,ROIs):
    """
    Parameters
    ----------
    pts : n x 2 ndarray
        Each row is a point (x,y)
    ROIs : List of ROIs, each ROI is a size 4 iterable
        Each ROI consists of (x1,y1,x2,y2), where (x1,y1) is the top left point
        and (x2,y2) is the bottom right point

    Returns
    -------
    ndarray of mask

    """
    submasks = []
    for ROI in ROIs:
        x_mask = np.logical_and(pts[:,0] >= ROI[0],pts[:,0] <= ROI[2])
        y_mask = np.logical_and(pts[:,1] >= ROI[1],pts[:,1] <= ROI[3])
        submasks.append(np.logical_and(x_mask,y_mask))
        
    final_mask = np.zeros(submasks[0].shape,dtype=bool)
    for mask in submasks:
        final_mask = np.logical_or(final_mask,mask)
        
    return final_mask

def drawPoints(img,pts):
    """
    Parameters
    ----------
    img : Image ndarray
        
    pts : n by 2 ndarray

    Returns
    -------
    img : Image ndarray
    """
    assert(pts.shape[1]==2)
    n_points = pts.shape[0]
    
    for i in range(n_points):
        pt = tuple(pts[i,:])
        img = cv.circle(img,pt,radius=5,color=(0,255,0))
    
    return img

def SIFT(img):
    descriptor = cv.SIFT_create()
    kps, des = descriptor.detectAndCompute(img, None)
    print("Number of features for SIFT:",len(kps))
    img_output =cv.drawKeypoints(img, kps, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_output

def SIFT_filter(kps, des, mask):
    kps_filtered = ()
    des_filtered = []
    for i in range(len(mask)):
        if mask[i] == 1:
            kps_filtered = kps_filtered + (kps[i],)
            des_filtered.append(des[i])
    
    des_filtered = np.asarray(des_filtered)
    
    return kps_filtered, des_filtered

    
if __name__ == '__main__':
    # Load exmaple image
    img = cv.imread("lamp_15.JPG")

    # show the original image and equalhist image
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img_histequal = equalizeHist(img)
    plt.imshow(cv.cvtColor(img_histequal, cv.COLOR_BGR2RGB))
    plt.axis('off')

    # Extract the features ORB
    kps, dps = findFeatures(img)
    kps_hist, dps_hist = findFeatures(img_histequal)
    print(type(kps_hist))
    print(dps_hist.shape)

    img_kps = cv.drawKeypoints(img, kps, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_hist_kps = cv.drawKeypoints(img_histequal, kps_hist, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    print("Number of SIFT features: ", len(kps_hist))


    plt.figure(2)
    plt.suptitle('SIFT Features Comparison')
    plt.subplot(1, 4, 1)
    plt.imshow(cv.cvtColor(img_kps, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv.cvtColor(img_hist_kps, cv.COLOR_BGR2RGB))
    plt.axis('off')   

    # Manual Set
    ROIs = np.array([[250,400,800,1000]])
    pts_hist = cv.KeyPoint_convert(kps_hist)
    final_mask = getMaskPointsInROIs(pts_hist,ROIs)
    pts_hist = pts_hist[final_mask]
    #kps_hist = cv.KeyPoint_convert(kps_hist)
    print("Number of Features: ",len(pts_hist))

    
    # Filter the selected features with information
    kps_filter,_= SIFT_filter(kps_hist, dps_hist,final_mask)
    print('The features filtered: ',len(kps_filter))
    img_hist_kps_filter = cv.drawKeypoints(img_histequal,kps_filter,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.subplot(1,4,3)
    plt.imshow(cv.cvtColor(img_hist_kps_filter, cv.COLOR_BGR2RGB))
    plt.axis('off')

    # Select the features lose information
    img_hist_ROI = drawPoints(img_histequal,pts_hist.astype(int))

    plt.subplot(1, 4, 4)
    plt.imshow(cv.cvtColor(img_hist_ROI, cv.COLOR_BGR2RGB))
    plt.axis('off')



    plt.show()




