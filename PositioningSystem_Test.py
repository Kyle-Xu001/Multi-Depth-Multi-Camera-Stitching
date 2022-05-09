import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torch import _batch_norm_impl_index
import PositioningSystem as ps
import transform
from utils import getParams
from stitch_custom import stitchImages


def getPosMatrix(img_ID, homo_params):
    '''
    Estimate the transformation matrix for translation positioning
    
    For lamp dictionary, input the stitching order of each camera.
    The function will go over stitching steps to output the final translation matrix.
    ------------------
    :param  img_ID: ID name of the camera
    :param  homo_params: homography matrices of all cameras
    
    :return  print the translation matrix for target camera 
    '''
    lampDict = {}
    '''Arie'''
    # lampDict['lamp01'] = ("lamp02-lamp01","lamp03-lamp02","lamp04-lamp03","lamp05-lamp04")
    # lampDict['lamp02'] = ("lamp03-lamp02","lamp04-lamp03","lamp05-lamp04")
    # lampDict['lamp03'] = ("lamp04-lamp03","lamp05-lamp04")
    # lampDict['lamp04'] = ("lamp05-lamp04",)
    
    
    '''office_farm'''
    # lampDict['lamp03'] = ("lamp02-lamp03","lamp01-lamp02","correction")
    # lampDict['lamp02'] = ("lamp02-lamp03","correction")
    # lampDict['lamp01'] = ("correction",)
    
    
    '''Mathe'''
    lampDict["lamp02"] = ("lamp02-lamp06-correction","lamp02-lamp06-shrink")
    lampDict['lamp03'] = ("lamp02-lamp03","lamp02-lamp06-correction","lamp02-lamp06-shrink")
    lampDict['lamp04'] = ("lamp03-lamp04","lamp02-lamp06-correction","lamp02-lamp06-shrink")
    lampDict['lamp05'] = ("lamp04-lamp05","lamp02-lamp06-correction","lamp02-lamp06-shrink")
    lampDict['lamp06'] = ("lamp05-lamp06","lamp02-lamp06-correction","lamp02-lamp06-shrink")
    
    lampDict['lamp07'] = ("lamp07-correction",)
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


def box2obb(box, ID, trans_params):
    '''Testing Function for transforming local boxes into global boxes'''
    obb = ps.getPos_box_array(ID, box, trans_params)
    
    obb = obb[0:4,:]
    obb = np.row_stack((obb, obb[0,:]))
    
    return obb



if __name__ == '__main__':

    '''Mathe Farm Positioning Test'''
    # Define the homography stitching parameters
    homo_params = getParams("params/homo_params_Mathe.json")
    for homo_param in homo_params:
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
     # Define the stitching arguments
    stitch_params = getParams("params/stitch_params_Mathe.json")
    
    # Define the translation parameters
    trans_params = getParams("params/trans_params_Mathe.json")
    
    # Define the dictionary for images
    imgs = {}
    imgs["lamp02"] = cv.imread("dataset/Mathe/lamp_02_Mathe.PNG")
    imgs["lamp03"] = cv.imread("dataset/Mathe/lamp_03_Mathe.PNG")
    imgs["lamp04"] = cv.imread("dataset/Mathe/lamp_04_Mathe.PNG")
    imgs["lamp05"] = cv.imread("dataset/Mathe/lamp_05_Mathe.PNG")
    imgs["lamp06"] = cv.imread("dataset/Mathe/lamp_06_Mathe.PNG")
    imgs["lamp07"] = cv.imread("dataset/Mathe/lamp_07_Mathe.PNG")
    imgs["lamp14"] = cv.imread("dataset/Mathe/lamp_14_Mathe.PNG")
    imgs["lamp15"] = cv.imread("dataset/Mathe/lamp_15_Mathe.PNG")
    imgs["lamp16"] = cv.imread("dataset/Mathe/lamp_16_Mathe.PNG")
    imgs["lamp17"] = cv.imread("dataset/Mathe/lamp_17_Mathe.PNG")
    imgs["lamp18"] = cv.imread("dataset/Mathe/lamp_18_Mathe.PNG")
    imgs["lamp19"] = cv.imread("dataset/Mathe/lamp_19_Mathe.PNG")
    imgs["lamp20"] = cv.imread("dataset/Mathe/lamp_20_Mathe.PNG")
    imgs["lamp21"] = cv.imread("dataset/Mathe/lamp_21_Mathe.PNG")
    imgs["lamp22"] = cv.imread("dataset/Mathe/lamp_22_Mathe.PNG")
    imgs["lamp23"] = cv.imread("dataset/Mathe/lamp_23_Mathe.PNG")
    
    # Stitch the Mathe Farm
    panorama = stitchImages(imgs, homo_params, stitch_params,'Mathe')    
    
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
           ['lamp20',(362, 568)],
           ['lamp07',(570, 230)],
           ['lamp07',(755, 230)],
           ['lamp07',(576, 562)],
           ['lamp06',(495, 215)],
           ['lamp05',(677, 187)],
           ['lamp04',(580, 320)],
           ['lamp03',(685, 350)],
           ['lamp02',(613, 547)]]
    
    # Define the transform boxes in original images
    box_15 = [[95, 291, 217, 242, 250, 300, 145, 350]]
    box_23 = [[386, 419, 620, 419, 620, 500, 386, 500]]
    box_07 = [[570, 230, 764, 230, 764, 570, 570, 570]]
    box_06 = [[90, 150, 300, 150, 300, 250, 90, 250]]
    box_05 = [[591, 295, 723, 295, 724, 401, 591, 400]]
    box_04 = [[477, 329, 641, 323, 640, 430, 477, 430]]
    box_03 = [[330, 405, 444, 405, 444, 600, 330, 600]]
    box_02 = [[550, 530, 685, 530, 685, 565, 550, 565]]
    
    obb1 = box2obb(box_15, 'lamp15', trans_params)
    obb2 = box2obb(box_23, 'lamp23', trans_params)
    obb3 = box2obb(box_07, 'lamp07', trans_params)
    obb4 = box2obb(box_06, 'lamp06', trans_params)
    obb5 = box2obb(box_05, 'lamp05', trans_params)
    obb6 = box2obb(box_04, 'lamp04', trans_params)
    obb7 = box2obb(box_03, 'lamp03', trans_params)
    obb8 = box2obb(box_02, 'lamp02', trans_params)
    
    # Get global points from each camera
    pts_global = []
    
    for pt in pts:
        pt_global = ps.getPos(pt[0],pt[1],trans_params)
        pts_global.append([pt_global[0], pt_global[1]])
        
    pts_global = np.array(pts_global)
    
    
    # Visualize the Transformation Result
    plt.figure(1)    
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.scatter(pts_global[:,0],pts_global[:,1],marker='+',color='r')
    plt.plot(obb1[:,0],obb1[:,1],color='r')
    plt.plot(obb2[:,0],obb2[:,1],color='r')
    plt.plot(obb3[:,0],obb3[:,1],color='r')
    plt.plot(obb4[:,0],obb4[:,1],color='r')
    plt.plot(obb5[:,0],obb5[:,1],color='r')
    plt.plot(obb6[:,0],obb6[:,1],color='r')
    plt.plot(obb7[:,0],obb7[:,1],color='r')
    plt.plot(obb8[:,0],obb8[:,1],color='r')
    plt.axis('off')
    
    
    
    
    '''Office Farm Positioning Test'''
    # Define the homography stitching parameters
    homo_params = getParams("params/homo_params_office_farm.json")
    for homo_param in homo_params:
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    # Define the stitching arguments
    stitch_params = getParams("params/stitch_params_office_farm.json")
    
    # Define the translation parameters
    trans_params = getParams("params/trans_params_office_farm.json")
        
    # Define the dictionary for images
    imgs = {}
    imgs["lamp01"] = cv.imread("dataset/office_farm/lamp_01_office.PNG")
    imgs["lamp02"] = cv.imread("dataset/office_farm/lamp_02_office.PNG")
    imgs["lamp03"] = cv.imread("dataset/office_farm/lamp_03_office.PNG")
    
    panorama = stitchImages(imgs, homo_params, stitch_params,'office_farm')
    
    # Define the transform point in original images
    pts = [['lamp01',(713, 230)],
           ['lamp01',(814, 70)],
           ['lamp02',(874, 260)],
           ['lamp02',(912, 309)],
           ['lamp03',(800, 560)],
           ['lamp03',(1200, 600)]]
    
    # Define Rectangle Boxs in lamp01
    boxes_01 = [[525, 80, 725, 80, 725, 600, 520, 590]]
    
    # Define Rectangle Boxs in lamp03
    boxes_03 = [[530, 400, 840, 400, 830, 555, 520, 550]]
    
    '''Test: Boxes Array in global image'''
    obb = ps.getPos_box_array('lamp01',boxes_01, trans_params)
    obb_ = ps.getPos_box_array('lamp03',boxes_03, trans_params)
    
    obb1 = obb[0:4,:]
    obb1 = np.row_stack((obb1, obb1[0,:]))
    
    obb3 = obb_[0:4,:]
    obb3 = np.row_stack((obb3, obb3[0,:]))
    
    
    '''Test: Point Transformation in global image'''
    pts_global = []
    
    for pt in pts:
        pt_global = ps.getPos(pt[0],pt[1],trans_params)
        pts_global.append([pt_global[0], pt_global[1]])
    
    pts_global = np.array(pts_global)


    plt.figure(2)
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.scatter(pts_global[:,0],pts_global[:,1],marker='+',color='r')
    plt.plot(obb1[:,0],obb1[:,1],color='r')
    plt.plot(obb3[:,0],obb3[:,1],color='r')
    plt.axis('off')
    
    
    
    
    '''Arie Farm Positioning Test'''
    # Define the homography stitching parameters
    homo_params = getParams("params/homo_params_Arie.json")
    for homo_param in homo_params:
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    # Define the stitching arguments
    stitch_params = getParams("params/stitch_params_Arie.json")
    
    # Define the translation parameters
    trans_params = getParams("params/trans_params_Arie.json")
   
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
    
    # Define Rectangle Boxs in lamp03
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


    plt.figure(3)
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.scatter(pts_global[:,0],pts_global[:,1],marker='+',color='r')
    plt.plot(obb1[:,0],obb1[:,1],color='r')
    plt.plot(obb2[:,0],obb2[:,1],color='r')
    plt.plot(obb3[:,0],obb3[:,1],color='r')
    plt.axis('off')
    plt.show()
