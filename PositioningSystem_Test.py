import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PositioningSystem as ps
import undistortion
import ImageStitch
import transform
from utils import getParams
from ImageStitch import Stitch
from stitch_custom import stitchImages



def getPosMatrix(img_ID, homo_params):
    '''Estimate the transformation matrix for point positioning'''
    
    lampDict = {}
    lampDict['lamp01'] = ("lamp02-lamp01","lamp03-lamp02","lamp04-lamp03","lamp05-lamp04")
    lampDict['lamp02'] = ("lamp03-lamp02","lamp04-lamp03","lamp05-lamp04")
    lampDict['lamp03'] = ("lamp04-lamp03","lamp05-lamp04")
    lampDict['lamp04'] = ("lamp05-lamp04",)
    
    lampDict['lamp14'] = ("lamp15-lamp14","lamp16-lamp15","lamp17-lamp16","lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp15'] = ("lamp16-lamp15","lamp17-lamp16","lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp16'] = ("lamp17-lamp16","lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp17'] = ("lamp18-lamp17","Right_Transform","stitch_total")
    lampDict['lamp18'] = ("Right_Transform","stitch_total")
    
    lampDict['lamp23'] = ("flip","lamp23-lamp22","lamp22-lamp21","lamp21-lamp20","lamp20-lamp19","Left_Transform")
    lampDict['lamp22'] = ("flip","lamp22-lamp21","lamp21-lamp20","lamp20-lamp19","Left_Transform")
    lampDict['lamp21'] = ("flip","lamp21-lamp20","lamp20-lamp19","Left_Transform")
    lampDict['lamp20'] = ("flip","lamp20-lamp19","Left_Transform")
    lampDict['lamp19'] = ("flip","Left_Transform")
    
    
    trans_mat = np.eye(3)
    for param in lampDict[img_ID]:
        trans_mat = np.matmul(homo_params[param], trans_mat)
    
    # Normalize the homography matrix
    trans_mat = trans_mat/trans_mat[2,2]
    
    np.set_printoptions(suppress=True)
    print(trans_mat.flatten().tolist())

    
if __name__ == '__main__':
    
    # homo_params = getParams("params/homo_params_Arie.json")
    # for homo_param in homo_params:
    #     homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    # stitch_params = getParams("params/stitch_params_Arie.json")
    
    # trans_params = getParams("params/trans_params_Arie.json")
    # for trans_param in trans_params:
    #     trans_params[trans_param] = np.array(trans_params[trans_param]).reshape(-1, 3)
    
    # # Define the dictionary for images
    # imgs = {}
    # imgs["lamp01"] = cv.imread("dataset/Arie/lamp_01_Arie.PNG")
    # imgs["lamp02"] = cv.imread("dataset/Arie/lamp_02_Arie.PNG")
    # imgs["lamp03"] = cv.imread("dataset/Arie/lamp_03_Arie.PNG")
    # imgs["lamp04"] = cv.imread("dataset/Arie/lamp_04_Arie.PNG")
    # imgs["lamp05"] = cv.imread("dataset/Arie/lamp_05_Arie.PNG")
    
    # panorama = stitchImages(imgs, homo_params, stitch_params,'Arie')
    
    # # load the initial images and corresponding homo matrix
    # img1_ = cv.imread("dataset/Paranoma/lamp_18_031513.PNG")
    # img2_ = cv.imread("dataset/Paranoma/lamp_17_031513.PNG")
    # img3_ = cv.imread("dataset/Paranoma/lamp_16_031513.PNG")
    # img4_ = cv.imread("dataset/Paranoma/lamp_15_031513.PNG")
    # img5_ = cv.imread("dataset/Paranoma/lamp_14_031513.PNG")

    # # Define the operating lamp
    # lamp_id1 = 'lamp18'
    # lamp_id2 = 'lamp17'
    # lamp_id3 = 'lamp16'
    # lamp_id4 = 'lamp15'
    # lamp_id5 = 'lamp14'
    
    # # Enter the direction of the parameters
    # calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"
    # farm_name = 'Mathe'

    # # Generate the undistorted image according to intrinsic parameters
    # img1, _, _ = undistortion.undistort(img1_, lamp_id1, calib_dir)
    # img2, _, _ = undistortion.undistort(img2_, lamp_id2, calib_dir)
    # img3, _, _ = undistortion.undistort(img3_, lamp_id3, calib_dir)
    # img4, _, _ = undistortion.undistort(img4_, lamp_id4, calib_dir)
    # img5, _, _ = undistortion.undistort(img5_, lamp_id5, calib_dir)
    
    # # Define the dictionary for images
    # imgs = {}
    # imgs["lamp14"] = img5
    # imgs["lamp15"] = img4
    # imgs["lamp16"] = img3
    # imgs["lamp17"] = img2
    # imgs["lamp18"] = img1
    
    # plt.figure(0)
    # plt.imshow(img4)
    # plt.axis('off')
    
    # # load the initial images and corresponding homo matrix
    # img1_ = cv.imread("dataset/Paranoma/lamp_23_031513.PNG")
    # img2_ = cv.imread("dataset/Paranoma/lamp_22_031513.PNG")
    # img3_ = cv.imread("dataset/Paranoma/lamp_21_031513.PNG")
    # img4_ = cv.imread("dataset/Paranoma/lamp_20_031513.PNG")
    # img5_ = cv.imread("dataset/Paranoma/lamp_19_031513.PNG")
    
    # # Define the operating lamp
    # lamp_id1 = 'lamp23'
    # lamp_id2 = 'lamp22'
    # lamp_id3 = 'lamp21'
    # lamp_id4 = 'lamp20'
    # lamp_id5 = 'lamp19'


    # # Calculate the mapping matrix
    # img1, _, _ = undistortion.undistort(img1_, lamp_id1, calib_dir)
    # img2, _, _ = undistortion.undistort(img2_, lamp_id2, calib_dir)
    # img3, _, _ = undistortion.undistort(img3_, lamp_id3, calib_dir)
    # img4, _, _ = undistortion.undistort(img4_, lamp_id4, calib_dir)
    # img5, _, _ = undistortion.undistort(img5_, lamp_id5, calib_dir)
    
    # imgs["lamp23"] = img1
    # imgs["lamp22"] = img2
    # imgs["lamp21"] = img3
    # imgs["lamp20"] = img4
    # imgs["lamp19"] = img5
    
    # panorama = stitchImages(imgs, homo_params, farm_name)
    
    
    # Define the transform point in original images
    # pts = [['lamp14',(710, 292)],
    #        ['lamp14',(415, 296)],
    #        ['lamp14',(216, 313)],
    #        ['lamp15',(220, 275)],
    #        ['lamp15',(580, 230)],
    #        ['lamp16',(400, 307)],
    #        ['lamp16',(720, 392)],
    #        ['lamp17',(398, 331)],
    #        ['lamp17',(561, 273)],
    #        ['lamp18',(560, 370)],
    #        ['lamp18',(560, 187)],
    #        ['lamp23',(500, 390)],
    #        ['lamp23',(685, 493)],
    #        ['lamp23',(684, 590)],
    #        ['lamp22',(673, 580)],
    #        ['lamp22',(675, 480)],
    #        ['lamp21',(635, 547)],
    #        ['lamp21',(692, 547)],
    #        ['lamp21',(749, 542)],
    #        ['lamp20',(362, 568)]]
    
    # box = [246, 225, 318, 250, 227, 375, 146, 330]
    # box_15 = [[95, 291, 217, 242, 250, 300, 145, 350],
    #           [140, 480, 220, 480, 180, 620, 100, 610]]
    
    # print(np.array(box_15).reshape(-1,2))
    
    # obb = getPos_box_array('lamp15',box_15, trans_params)
    # obb1 = obb[0:4,:]
    # obb1 = np.row_stack((obb1, obb1[0,:]))
    
    # obb2= obb[4:8,:]
    # obb2 = np.row_stack((obb2, obb2[0,:]))
    
    # pts_global = []
    
    # for pt in pts:
    #     pt_global = getPos(pt[0],pt[1],trans_params)
    #     pts_global.append([pt_global[0], pt_global[1]])
    
    # pts_global = np.array(pts_global)
    
    '''Arie Farm Positioning Test'''
    # Define the homography stitching parameters
    homo_params = getParams("params/homo_params_Arie.json")
    for homo_param in homo_params:
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    # Define the stitching arguments
    stitch_params = getParams("params/stitch_params_Arie.json")
    
    # Define the translation parameters
    trans_params = getParams("params/trans_params_Arie.json")
    for trans_param in trans_params:
        trans_params[trans_param] = np.array(trans_params[trans_param]).reshape(-1, 3)
    
    # Define the dictionary for images
    imgs = {}
    imgs["lamp01"] = cv.imread("dataset/Arie/lamp_01_Arie.PNG")
    imgs["lamp02"] = cv.imread("dataset/Arie/lamp_02_Arie.PNG")
    imgs["lamp03"] = cv.imread("dataset/Arie/lamp_03_Arie.PNG")
    imgs["lamp04"] = cv.imread("dataset/Arie/lamp_04_Arie.PNG")
    imgs["lamp05"] = cv.imread("dataset/Arie/lamp_05_Arie.PNG")
    
    panorama = stitchImages(imgs, homo_params, stitch_params,'Arie')
    
    # Define the transform point in original images
    pts = [['lamp01',(385, 388)],
           ['lamp01',(427, 576)],
           ['lamp01',(615, 122)],
           ['lamp02',(639, 283)],
           ['lamp02',(468, 272)],
           ['lamp03',(612, 244)],
           ['lamp03',(529, 324)],
           ['lamp04',(650, 144)],
           ['lamp04',(518, 139)],
           ['lamp05',(554, 299)]]
    
    # Define Rectangle Boxs in lamp01
    boxes_01 = [[390, 300, 535, 295, 535, 385, 390, 380],
                [560, 140, 700, 140, 700, 215, 560, 215]]
    
    # Define Rectangle Boxs in lamp01
    boxes_03 = [[435, 240, 747, 249, 743, 336, 433, 334]]
    
    '''Test: Boxes Array in global image'''
    obb = ps.getPos_box_array('lamp01',boxes_01, trans_params)
    obb_ = ps.getPos_box_array('lamp03',boxes_03, trans_params)
    
    obb1 = obb[0:4,:]
    obb1 = np.row_stack((obb1, obb1[0,:]))
    
    obb2 = obb[4:8,:]
    obb2 = np.row_stack((obb2, obb2[0,:]))
    
    obb3 = obb_[0:4,:]
    obb3 = np.row_stack((obb3, obb3[0,:]))
    
    
    '''Test: Point Transformation in global image'''
    pts_global = []
    
    for pt in pts:
        pt_global = ps.getPos(pt[0],pt[1],trans_params)
        pts_global.append([pt_global[0], pt_global[1]])
    
    pts_global = np.array(pts_global)


    plt.figure(1)
    #plt.imshow(cv.cvtColor(imgs['lamp03'], cv.COLOR_BGR2RGB))
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.scatter(pts_global[:,0],pts_global[:,1],marker='+',color='r')
    plt.plot(obb1[:,0],obb1[:,1],color='r')
    plt.plot(obb2[:,0],obb2[:,1],color='r')
    plt.plot(obb3[:,0],obb3[:,1],color='r')
    plt.axis('off')
    plt.show()
