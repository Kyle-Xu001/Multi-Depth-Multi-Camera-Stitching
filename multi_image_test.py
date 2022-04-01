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
    # homo_mat = np.array([[
        # 0.9962750152836561, 0.011394294761734253, -15.863355184114155,
        # -0.005241890227618538, 1.0267777741157014, 411.4732629218854,
        # -0.000006746897821467397, 0.00004496214822227408, 1.0]]).reshape(-1, 3)
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
        1.010638759789845, -0.020425372912002478, -14.695282380876835,
        0.0075814846672632484, 1.0066178703225623, 412.43074450438576,
        2.1888462791375407e-05, -2.2383079441019405e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img4, img5, homo_mat)

    homo_mat = np.array([[0.9111138797319636, 0.025992907667642014, 43.51630224609544, -0.08061531667103257, 0.976052629267486, 444.77675271224103, -7.790215434632145e-05, 2.9319041438978976e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img3, img_stitch, homo_mat)

    homo_mat = np.array([[0.9296608798773768, -0.04505056659216292, 78.06623318399255, -0.016900992281276207, 0.917599731615178, 400.8510671764359, -3.617106314861764e-05, -6.205541847742475e-05, 1.0]]).reshape(-1, 3)
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
    pts = np.array([[200, 0],
                    [850, 0],
                    [700, 1800],
                    [225, 2250]])
    
    dst = np.array([[180, 0],
                    [850, 0],
                    [730, 1850],
                    [380, 2150]], dtype="float32")

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
    homo_mat = np.array([[1.0223869691572842, -0.029757793305358708, -9.554931168189306,
                          -0.10257780654402092, 0.9829667304728431, 382.29502902382444,
                          2.9003442218944775e-05, -4.261237504410896e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img2, img1, homo_mat)

    homo_mat = np.array([[0.895863627101159, -0.009314220498154528, 39.69811368863527,
                          -0.14319312890711713, 0.9238744611145023, 407.6464453689328, 
                          -0.00010585661979243849, 1.1089445174037917e-05, 0.9999999999999999]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img3, img_stitch, homo_mat)

    homo_mat = np.array([[1.0093568603313154, -0.024777213229165367, -1.7110387731670804,
                          -0.0917809278675415, 0.9321125574704138, 497.27554411496135,
                          3.0226341371213186e-05, -4.5460503076694054e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img4, img_stitch, homo_mat)

    homo_mat = np.array([[1.136892310007503, -0.030797753772207552, -25.882898617872147,
                          -0.08009097612575575, 1.0539222856379677, 474.91596299428886,
                          0.00012140360121091282, -3.315567108552388e-05, 1.0]]).reshape(-1, 3)
    img_stitch = ImageStitch.simpleStitch(img5, img_stitch, homo_mat)

    img_stitch = cv.flip(img_stitch, 0)

    plt.figure(2)
    plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    '''Transform Stitching Result to avoid Distortion in final stitching'''
    bottom = img_stitch.shape[0]
    pts = np.array([[180, 400],
                    [930, 910],
                    [920, bottom],
                    [270, bottom]])
    
    dst = np.array([[250, 775],
                    [910, 775],
                    [920, bottom],
                    [270, bottom]], dtype="float32")

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
    # cv.namedWindow("Area Selection", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Area Selection", 800, 600)
    
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
    homo_mat = np.array([[0.9131002758816589, -0.10864560677082075, 108.34106936850189, -0.22084159252494037, 0.7010361167342065, 2494.7795892747777, -3.353712513337784e-05, -0.00011457038122211423, 1.0]]).reshape(-1, 3)

    #img_stitch = cv.flip(img_stitch, 0)
    img_stitch = ImageStitch.simpleStitch(img_stitch, img_stitch_, homo_mat)
    paranoma = img_stitch[500:5300,50:1450,:]
    
    plt.figure(5)
    plt.imshow(cv.cvtColor(np.rot90(paranoma), cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    '''
    Print the parameters of homography matrix
    '''
    np.set_printoptions(suppress=True)
    print(homo_mat.flatten().tolist())