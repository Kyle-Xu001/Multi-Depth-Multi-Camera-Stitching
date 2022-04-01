import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import transform
import ImageStitch
import undistortion
from ImageStitch import Stitch


# Define the main function
if __name__ == '__main__':
    # Define the draw parameters for matching visualization
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(0, 0, 255),
                       flags=cv.DrawMatchesFlags_DEFAULT)

    '''
    Stitch the Right Area from lamp14 to lamp18
    
    Generate estimated homography matrix using matches between corresponding features in ROIs.
    Directly apply homography matrix to stitch all frames
    --------------------
    
    :param img: images from each frame
    :type img: nparray (768*1152*3)
    
    :param homo_mat: homography matrix generated from feature estimation, using for image transformation
    :type homo_mat: nparray (3*3)
    
    :return img_stitch: stitching image of lamp14 to lamp18
    '''
    
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

    '''Stitch Images'''
    # Param Tesing 1 
    # homo_mat = np.array([[1.010638759789845, -0.020425372912002478, -14.695282380876835,
    #                       0.0075814846672632484, 1.0066178703225623, 412.43074450438576,
    #                       2.1888462791375407e-05, -2.2383079441019405e-05, 1.0]]).reshape(-1, 3)
    # img_stitch = ImageStitch.simpleStitch(img4, img5, homo_mat)

    # homo_mat = np.array([[0.9058392172187458, 0.0004990937109045671, 47.304134684316686,
    #                       -0.06666970815387915, 0.9191001811983562, 444.4072208488787,
    #                       -0.00006927906859194502, -0.00004704323491312332,1.0]]).reshape(-1, 3)
    # img_stitch = ImageStitch.simpleStitch(img3, img_stitch, homo_mat)

    # homo_mat = np.array([[0.955857230597833, -0.057484217402180626, 71.90318975556875,
    #                       -0.007353288099102101, 0.8925199803621107, 405.46235944750543,
    #                       -0.00001106889426245175, -0.00008664024139811333, 1.0]]).reshape(-1, 3)
    # img_stitch = ImageStitch.simpleStitch(img2, img_stitch, homo_mat)

    # homo_mat = np.array([[1.10662515274729, -0.026531652925349116, -94.98865905700784,
    #                       0.052902376594418724, 1.0362753243612786, 433.98218868249035,
    #                       0.00008008349410686805, -0.000001116226659156957,1.0]]).reshape(-1, 3)
    # img_stitch_ = ImageStitch.simpleStitch(img1, img_stitch, homo_mat)
    
    # Param Tesing 2
    homo_mat = np.array([[
        0.9962750152836561, 0.011394294761734253, -15.863355184114155,
        -0.005241890227618538, 1.0267777741157014, 411.4732629218854,
        -0.000006746897821467397, 0.00004496214822227408, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img4, img5, homo_mat)

    homo_mat = np.array([[
        0.8625498230406553, -0.02301907588895937, 59.70997505303934,
        -0.08656426471273822, 0.8907494006294058, 441.9432628853975,
        -0.00011029468806013377, -0.00004138440551081008, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img3, img_stitch, homo_mat)

    homo_mat = np.array([[
        0.9448958866207864, -0.042861248612460975, 72.84439636268718,
        -0.013266144059998328, 0.9300566760247033, 399.48686600411634,
        -0.000025221215424675617, -0.00006304664715663776, 0.9999999999999999]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img2, img_stitch, homo_mat)

    homo_mat = np.array([[
        1.1386527445204575, 0.031927187738166204, -109.45549680535996,
        0.04498832067998204, 1.0320031734670985, 463.08525965607436,
        0.00010009854442945858, 0.00006315329701074331, 1.0 ]]).reshape(-1, 3)
    img_stitch_ = ImageStitch.simpleStitch(img1, img_stitch, homo_mat)

    plt.figure(0)
    plt.imshow(cv.cvtColor(img_stitch_, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    '''Transform Stitching Result to avoid Distortion in final stitching'''
    pts = np.array([[0, 0],
                    [1100, 0],
                    [1200, 2500],
                    [50, 2550]])
    
    dst = np.array([[0, 0],
                    [1100, 0],
                    [1250, 2800],
                    [250, 2200]], dtype="float32")

    img_stitch_ = transform.four_point_transform(img_stitch_, pts, dst)

    plt.figure(1)
    plt.imshow(cv.cvtColor(np.rot90(img_stitch_), cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    # '''Using feature-based estimation to generate homography matrix'''
    # # load the matching images
    # img1_ = cv.imread("dataset/lamp1415161718-lamp19/lamp_19_031613.PNG")
    # lamp_id1 = 'lamp19'
    
    # img1, _, _ = undistortion.undistort(img1_, lamp_id1, calib_dir)
    # img2 = img_stitch

    # ROIs1 = cv.selectROIs("Area Selection", img1)
    # ROIs2 = cv.selectROIs("Area Selection", img2)

    # for i in range(len(ROIs1)):
    #     ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
    #     ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
    #     ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
    #     ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]

    # stitches = Stitch(img1, img2)
    # stitches.featureExtract(ROIs1, ROIs2)
    # homo_mat, matches_inliers = stitches.homoEstimate()
    # kps1 = stitches.Img1.kps
    # kps2 = stitches.Img2.kps

    # # Visualization
    # print("\nNumber of inlier matches: ", len(matches_inliers), "\n")
    # img_inliers = cv.drawMatches(
    #     img1, kps1, img2, kps2, matches_inliers, None, **draw_params)

    # plt.figure(2)
    # plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
    # plt.title("Inlier Matches for Total Selected Area [Num: %d]"%(len(matches_inliers)))
    # plt.axis('off')

    # '''Stitch the Images'''
    # img_stitch_ = ImageStitch.simpleStitch(img1, img2, homo_mat)

    # plt.figure(3)
    # plt.imshow(cv.cvtColor(np.rot90(img_stitch_), cv.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    '''
    Stitch the Left Area from lamp23 to lamp19
    
    Generate estimated homography matrix using matches between corresponding features in ROIs.
    Directly apply homography matrix to stitch all frames
    --------------------
    
    :param img: images from each frame
    :type img: nparray (768*1152*3)
    
    :param homo_mat: homography matrix generated from feature estimation, using for image transformation
    :type homo_mat: nparray (3*3)
    
    :return img_stitch: stitching image of lamp14 to lamp18
    '''
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

    # Enter the direction of the parameters
    calib_dir = "/home/cxu-lely/kyle-xu001/Multi-Depth-Multi-Camera-Stitching/calib_params_Mathe"

    # Calculate the mapping matrix
    img1, _, _ = undistortion.undistort(img1_, lamp_id1, calib_dir)
    img2, _, _ = undistortion.undistort(img2_, lamp_id2, calib_dir)
    img3, _, _ = undistortion.undistort(img3_, lamp_id3, calib_dir)
    img4, _, _ = undistortion.undistort(img4_, lamp_id4, calib_dir)
    img5, _, _ = undistortion.undistort(img5_, lamp_id5, calib_dir)

    # Keep the img5 as background image, stitch from left to right
    img1 = cv.flip(img1, 0)
    img2 = cv.flip(img2, 0)
    img3 = cv.flip(img3, 0)
    img4 = cv.flip(img4, 0)
    img5 = cv.flip(img5, 0)

    '''Stitch Images'''
    homo_mat = np.array([[1.0218209881498754, -0.042645259634918185, -7.994198996506845,
                          -0.10410234684075273, 0.966529343724799, 384.2736252631494,
                          3.3695504080035906e-05, -6.383390869394006e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img2, img1, homo_mat)

    homo_mat = np.array([[0.8947066148388445, 0.010700350267399688, 38.93350954074094,
                          -0.16533197135329256, 0.9464658136944826, 415.4927778218402,
                          -0.0001185577427515866, 7.046682325598091e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img3, img_stitch, homo_mat)

    homo_mat = np.array([[1.0603014253456642, -0.032790461947474384, -15.205888258186151,
                          -0.07755178218753744, 0.9310102440849826, 500.5701011114284,
                          7.265593193259331e-05, -6.169149879685164e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img4, img_stitch, homo_mat)

    homo_mat = np.array([[0.9774929391602935, -0.011050412845673601, 17.64029560530821,
                          -0.13494378393269393, 1.0111083058712496, 477.48252360100406,
                          -2.625850345364947e-06, -8.670763250018663e-06, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img5, img_stitch, homo_mat)

    img_stitch = cv.flip(img_stitch, 0)

    plt.figure(2)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    '''Transform Stitching Result to avoid Distortion in final stitching'''
    pts = np.array([[0, 50],
                    [1350, 700],
                    [1200, 2540],
                    [250, 2540]])
    
    dst = np.array([[0, 550],
                    [1350, 600],
                    [1200, 2540],
                    [200, 2540]], dtype="float32")

    img_stitch = transform.four_point_transform(img_stitch, pts, dst)
    
    plt.figure(3)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # '''Estimate the Homo Matrix for Final Stitching'''
    # # load the matching images
    # img1 = img_stitch
    # img2 = img_stitch_
    
    # # Define the size of opencv window
    # cv.namedWindow("select the area", cv.WINDOW_NORMAL)
    # cv.resizeWindow("select the area", 800, 600)
    
    # ROIs1 = cv.selectROIs("Area Selection", img1)
    # ROIs2 = cv.selectROIs("Area Selection", img2)

    # for i in range(len(ROIs1)):
    #     ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
    #     ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
    #     ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
    #     ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]

    # stitches = Stitch(img1, img2)
    # stitches.featureExtract(ROIs1, ROIs2)
    # homo_mat, matches_inliers = stitches.homoEstimate()
    # kps1 = stitches.Img1.kps
    # kps2 = stitches.Img2.kps

    # # Visualization
    # print("\nNumber of inlier matches: ", len(matches_inliers), "\n")
    # img_inliers = cv.drawMatches(
    #     img1, kps1, img2, kps2, matches_inliers, None, **draw_params)

    # plt.figure(4)
    # plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
    # plt.title("Inlier Matches for Total Selected Area [Num: %d]"%(len(matches_inliers)))
    # plt.axis('off')

    # '''Final Stitch'''
    # img_stitch = ImageStitch.simpleStitch(img1, img2, homo_mat)
    
    
    '''Final Stitch between two Area'''
    homo_mat = np.array([[1.531377223673118, -0.09420641594437201, -17.083883674895635,
                          1.0214203951714997, 1.0746240680265424, 2188.8250559369794,
                          0.00047561484942954256, -7.001838036624984e-05, 0.9999999999999999]]).reshape(-1, 3)

    #img_stitch = cv.flip(img_stitch, 0)
    img_stitch = ImageStitch.simpleStitch(img_stitch, img_stitch_, homo_mat)

    plt.figure(5)
    plt.imshow(cv.cvtColor(np.rot90(img_stitch), cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    '''
    Print the parameters of homography matrix
    '''
    np.set_printoptions(suppress=True)
    print(homo_mat.flatten().tolist())