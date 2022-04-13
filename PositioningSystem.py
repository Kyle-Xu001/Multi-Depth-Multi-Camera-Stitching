import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import undistortion
import ImageStitch
import transform
from utils import getHomoParams
from ImageStitch import Stitch
from stitch_custom import stitchImages


def getPosMatrix(img_ID, homo_params):
    '''Estimate the transformation matrix for point positioning'''
    
    lampDict = {}
    lampDict['lamp14'] = ("lamp15-lamp14","lamp16-lamp15","lamp17-lamp16","lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp15'] = ("lamp16-lamp15","lamp17-lamp16","lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp16'] = ("lamp17-lamp16","lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp17'] = ("lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp18'] = ("Right_Transform","stitch_total")
    
    lampDict['lamp23'] = ("flip","lamp23-lamp22","lamp22-lamp21","lamp21-lamp20","lamp20-lamp19","Left_Transform")
    lampDict['lamp22'] = ("flip","lamp22-lamp21","lamp21-lamp20","lamp20-lamp19","Left_Transform")
    lampDict['lamp21'] = ("flip","lamp21-lamp20","lamp20-lamp19","Left_Transform")
    lampDict['lamp20'] = ("flip","lamp20-lamp19","Left_Transform")
    lampDict['lamp19'] = ("flip","Left_Transform",)
    
    
    homo_mat = np.eye(3)
    for param in lampDict[img_ID]:
        homo_mat = np.matmul(homo_params[param], homo_mat)
    
    # Normalize the homography matrix
    homo_mat = homo_mat/homo_mat[2,2]
    
    np.set_printoptions(suppress=True)
    print(homo_mat.flatten().tolist())
    


def getPos(img_ID, pt, trans_params):
    '''get the global position of point from individual camera'''
    
    trans_param = trans_params[img_ID]
    pos = np.array(pt,dtype='float64').reshape([1,1,2])
    
    if img_ID == 'lamp14' or img_ID == 'lamp15' or img_ID == 'lamp16' or img_ID == 'lamp17' or img_ID == 'lamp18':
        # pos_new = np.matmul(trans_param, pos)
        # pos_new = pos_new/pos_new[2]
        pos_new = cv.perspectiveTransform(pos, trans_param).flatten()
        pt_new = (pos_new[0]-50, pos_new[1]-500)
    else:
        # pos_new = np.matmul(trans_param, pos)
        # pos_new = pos_new/pos_new[2]
        pos_new = cv.perspectiveTransform(pos, trans_param).flatten()
        pt_new = (pos_new[0]-50, 2303-pos_new[1])
    
    return pt_new
    
    
def getPos_box(img_ID, obb, trans_params):
    '''transform the single box to the panorama image'''
    trans_param = trans_params[img_ID]
    obb = np.asarray(obb,dtype='float64')
    obb = obb.reshape([1,int(len(obb)/2),2])
    
    obb_transform = cv.perspectiveTransform(obb, trans_param)
    obb_transform = obb_transform.reshape(obb.shape[1:])
    
    obb_transform = obb_transform + np.array([-50, -500])
    return obb_transform
    
    
if __name__ == '__main__':
    
    homo_params = getHomoParams("multi_stitch_params.json")
    for homo_param in homo_params:
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    trans_params = getHomoParams("positioning_params.json")
    for trans_param in trans_params:
        trans_params[trans_param] = np.array(trans_params[trans_param]).reshape(-1, 3)
    
    # load the initial images and corresponding homo matrix
    img1_ = cv.imread("dataset/Paranoma/lamp_18_031513.PNG")
    img2_ = cv.imread("dataset/Paranoma/lamp_17_031513.PNG")
    img3_ = cv.imread("dataset/Paranoma/lamp_16_031513.PNG")
    img4_ = cv.imread("dataset/Paranoma/lamp_15_031513.PNG")
    img5_ = cv.imread("dataset/Paranoma/lamp_14_031513.PNG")

    # Define the operating lamp
    lamp_id1 = 'lamp18'
    lamp_id2 = 'lamp17'
    lamp_id3 = 'lamp16'
    lamp_id4 = 'lamp15'
    lamp_id5 = 'lamp14'
    
    # Enter the direction of the parameters
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"

    # Generate the undistorted image according to intrinsic parameters
    img1, _, _ = undistortion.undistort(img1_, lamp_id1, calib_dir)
    img2, _, _ = undistortion.undistort(img2_, lamp_id2, calib_dir)
    img3, _, _ = undistortion.undistort(img3_, lamp_id3, calib_dir)
    img4, _, _ = undistortion.undistort(img4_, lamp_id4, calib_dir)
    img5, _, _ = undistortion.undistort(img5_, lamp_id5, calib_dir)
    
    # Define the dictionary for images
    imgs = {}
    imgs["lamp14"] = img5
    imgs["lamp15"] = img4
    imgs["lamp16"] = img3
    imgs["lamp17"] = img2
    imgs["lamp18"] = img1
    
    plt.figure(0)
    plt.imshow(img4)
    plt.axis('off')
    
    # load the initial images and corresponding homo matrix
    img1_ = cv.imread("dataset/Paranoma/lamp_23_031513.PNG")
    img2_ = cv.imread("dataset/Paranoma/lamp_22_031513.PNG")
    img3_ = cv.imread("dataset/Paranoma/lamp_21_031513.PNG")
    img4_ = cv.imread("dataset/Paranoma/lamp_20_031513.PNG")
    img5_ = cv.imread("dataset/Paranoma/lamp_19_031513.PNG")
    
    # Define the operating lamp
    lamp_id1 = 'lamp23'
    lamp_id2 = 'lamp22'
    lamp_id3 = 'lamp21'
    lamp_id4 = 'lamp20'
    lamp_id5 = 'lamp19'


    # Calculate the mapping matrix
    img1, _, _ = undistortion.undistort(img1_, lamp_id1, calib_dir)
    img2, _, _ = undistortion.undistort(img2_, lamp_id2, calib_dir)
    img3, _, _ = undistortion.undistort(img3_, lamp_id3, calib_dir)
    img4, _, _ = undistortion.undistort(img4_, lamp_id4, calib_dir)
    img5, _, _ = undistortion.undistort(img5_, lamp_id5, calib_dir)
    
    imgs["lamp23"] = img1
    imgs["lamp22"] = img2
    imgs["lamp21"] = img3
    imgs["lamp20"] = img4
    imgs["lamp19"] = img5
    
    panorama = stitchImages(imgs, homo_params)
    
    
    # Define the transform point in original images
    pts = [['lamp14',(710, 292)],
           ['lamp14',(415, 296)],
           ['lamp14',(216, 313)],
           ['lamp15',(220, 275)],
           ['lamp15',(580, 230)],
           ['lamp16',(400, 307)],
           ['lamp16',(720, 392)],
           ['lamp17',(398, 331)],
           ['lamp17',(561, 273)],
           ['lamp18',(560, 370)],
           ['lamp18',(560, 187)],
           ['lamp23',(500, 390)],
           ['lamp23',(685, 493)],
           ['lamp23',(684, 590)],
           ['lamp22',(673, 580)],
           ['lamp22',(675, 480)],
           ['lamp21',(635, 547)],
           ['lamp21',(692, 547)],
           ['lamp21',(749, 542)],
           ['lamp20',(362, 568)]]
    
    box = [246, 225, 318, 250, 227, 375, 146, 330]
    box_15 = [95, 291, 217, 242, 250, 300, 145, 350]
    
    obb = getPos_box('lamp15',box_15, trans_params)
    obb = np.row_stack((obb, obb[0,:]))
    
    pts_global = []
    
    for pt in pts:
        pt_global = getPos(pt[0],pt[1],trans_params)
        pts_global.append([pt_global[0], pt_global[1]])
    
    pts_global = np.array(pts_global)

    plt.figure(1)
    #plt.imshow(cv.cvtColor(np.rot90(paranoma), cv.COLOR_BGR2RGB))
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.scatter(pts_global[:,0],pts_global[:,1],marker='+',color='r')
    plt.plot(obb[:,0],obb[:,1],color='r')
    plt.axis('off')
    plt.show()
