import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from image_feature_extraction_test import Image
import image_match_test as match_utils

if __name__ == '__main__':
        # Define the draw parameters for matching visualization
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (0,0,255),
                    flags = cv.DrawMatchesFlags_DEFAULT)
        # load the matching images
        img1 = cv.imread("dataset/origin_images/lamp_15.JPG")
        img2 = cv.imread("dataset/origin_images/lamp_14.JPG")

        homo_mat = np.array([[
                0.9962750152836561,
                0.011394294761734253,
                -15.863355184114155,
                -0.005241890227618538,
                1.0267777741157014,
                411.4732629218854,
                -0.000006746897821467397,
                0.00004496214822227408,
                1.0
            ]]).reshape(-1,3)
        print(homo_mat)
        '''
        Stitch the Images
        '''
        # Get the position of vertices
        posVerts = utils.transformVerts(img_size=np.array([img1.shape[1],img1.shape[0]]), homo_mat=homo_mat)
        # print("Left Top: ",posVerts[0,:],"\n",
        #       "Right Top: ",posVerts[1,:],"\n",
        #       "Right Bottom: ",posVerts[2,:],"\n",
        #       "Left Bottom: ",posVerts[3,:],"\n")
                
        x_min = posVerts[:,0].min()
        x_max = posVerts[:,0].max()
        y_min = posVerts[:,1].min()
        y_max = posVerts[:,1].max()
        print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %(x_min,x_max,y_min,y_max))

        stitch_size = (x_max,y_max)

        homo_mat_ = np.eye(3)
        img_super = cv.warpPerspective(img1, homo_mat_,stitch_size,borderValue=(0,0,0))
        img_transform = cv.warpPerspective(img2, homo_mat,stitch_size,borderValue=(0,0,0))

        # Combine the image on one super image
        high_y = np.min(posVerts[:,1])
        img_transform[high_y:high_y,:,:] = 0
        img_super[img_transform>0]=0

        img_stitch = img_transform + img_super

        plt.figure(1)
        plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        # load the matching images
        img1 = cv.imread("dataset/origin_images/lamp_16.JPG")
        img2 = img_stitch

        ROIs1 = cv.selectROIs("select the area", img1)
        ROIs2 = cv.selectROIs("select the area", img2)

        for i in range(len(ROIs1)):
                ROIs1[i,2] = ROIs1[i,0] + ROIs1[i,2]
                ROIs1[i,3] = ROIs1[i,1] + ROIs1[i,3]
                ROIs2[i,2] = ROIs2[i,0] + ROIs2[i,2]
                ROIs2[i,3] = ROIs2[i,1] + ROIs2[i,3]

        # Initialize the object
        Img1 = Image(img1)
        Img2 = Image(img2)

        # Extract the features from each images
        kps1, des1 = Img1.findFeatures('sift')
        kps2, des2 = Img2.findFeatures('sift')   

        # Extract the masks to filter the features into several clusters
        masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
        masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)

        kpsCluster1, desCluster1 = Img1.featureCluster(masks1)
        kpsCluster2, desCluster2 = Img2.featureCluster(masks2)

        # Match the features with corresponding clusters in each image
        matches = utils.clusterMatch(desCluster1,desCluster2)


        # Integrate the clusters into one list
        kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)

        '''
        Find the parameters for homography matrix
        '''
        # Calculate the homography matrix for image transformation
        homo_mat, inliers_mask = utils.findHomography(matches, kps1_filter, kps2_filter)
        matches_inliers = list(itertools.compress(matches, inliers_mask))
        img_inliers = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers,None,**draw_params)


        print("\nNumber of inlier matches: ", len(matches_inliers),"\n")


        plt.figure(2)

        plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
        plt.title("Inlier Matches for Total Selected Area")
        plt.axis('off')


        # # Check the specific inliers
        # inlierCheck = input('Please Input [yes] to start checking the inliers:')
        # if inlierCheck == "yes":
        #     inliernum = int(input('Please Input the inlier Number you want to check:'))
        #     utils.inlierChecker(Img1,kps1_filter,Img2,kps2_filter,matches_inliers,draw_params,num=inliernum)
        

        '''
        Stitch the Images
        '''
        # Get the position of vertices
        posVerts = utils.transformVerts(img_size=np.array([Img1.img.shape[1],Img1.img.shape[0]]), homo_mat=homo_mat)
        # print("Left Top: ",posVerts[0,:],"\n",
        #       "Right Top: ",posVerts[1,:],"\n",
        #       "Right Bottom: ",posVerts[2,:],"\n",
        #       "Left Bottom: ",posVerts[3,:],"\n")
        
        x_min = posVerts[:,0].min()
        x_max = posVerts[:,0].max()
        y_min = posVerts[:,1].min()
        y_max = posVerts[:,1].max()
        print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %(x_min,x_max,y_min,y_max))

        stitch_size = (x_max,y_max)

        homo_mat_ = np.eye(3)
        img_super = cv.warpPerspective(Img1.img, homo_mat_,stitch_size,borderValue=(0,0,0))
        img_transform = cv.warpPerspective(Img2.img, homo_mat,stitch_size,borderValue=(0,0,0))

        # Combine the image on one super image
        high_y = np.min(posVerts[:,1])
        img_transform[high_y:high_y,:,:] = 0
        img_super[img_transform>0]=0

        img_stitch = img_transform + img_super

        plt.figure(3)
        plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.show()

         # load the matching images
        img1 = cv.imread("dataset/origin_images/lamp_17.JPG")
        img2 = img_stitch

        ROIs1 = cv.selectROIs("select the area", img1)
        ROIs2 = cv.selectROIs("select the area", img2)

        for i in range(len(ROIs1)):
                ROIs1[i,2] = ROIs1[i,0] + ROIs1[i,2]
                ROIs1[i,3] = ROIs1[i,1] + ROIs1[i,3]
                ROIs2[i,2] = ROIs2[i,0] + ROIs2[i,2]
                ROIs2[i,3] = ROIs2[i,1] + ROIs2[i,3]

        # Initialize the object
        Img1 = Image(img1)
        Img2 = Image(img2)

        # Extract the features from each images
        kps1, des1 = Img1.findFeatures('sift')
        kps2, des2 = Img2.findFeatures('sift')   

        # Extract the masks to filter the features into several clusters
        masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
        masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)

        kpsCluster1, desCluster1 = Img1.featureCluster(masks1)
        kpsCluster2, desCluster2 = Img2.featureCluster(masks2)

        # Match the features with corresponding clusters in each image
        matches = utils.clusterMatch(desCluster1,desCluster2)


        # Integrate the clusters into one list
        kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)

        '''
        Find the parameters for homography matrix
        '''
        # Calculate the homography matrix for image transformation
        homo_mat, inliers_mask = utils.findHomography(matches, kps1_filter, kps2_filter)
        matches_inliers = list(itertools.compress(matches, inliers_mask))
        img_inliers = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers,None,**draw_params)


        print("\nNumber of inlier matches: ", len(matches_inliers),"\n")


        plt.figure(2)

        plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
        plt.title("Inlier Matches for Total Selected Area")
        plt.axis('off')


        # # Check the specific inliers
        # inlierCheck = input('Please Input [yes] to start checking the inliers:')
        # if inlierCheck == "yes":
        #     inliernum = int(input('Please Input the inlier Number you want to check:'))
        #     utils.inlierChecker(Img1,kps1_filter,Img2,kps2_filter,matches_inliers,draw_params,num=inliernum)
        

        '''
        Stitch the Images
        '''
        # Get the position of vertices
        posVerts = utils.transformVerts(img_size=np.array([Img1.img.shape[1],Img1.img.shape[0]]), homo_mat=homo_mat)
        # print("Left Top: ",posVerts[0,:],"\n",
        #       "Right Top: ",posVerts[1,:],"\n",
        #       "Right Bottom: ",posVerts[2,:],"\n",
        #       "Left Bottom: ",posVerts[3,:],"\n")
        
        x_min = posVerts[:,0].min()
        x_max = posVerts[:,0].max()
        y_min = posVerts[:,1].min()
        y_max = posVerts[:,1].max()
        print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %(x_min,x_max,y_min,y_max))

        stitch_size = (x_max+500,y_max+500)

        homo_mat_ = np.eye(3)
        img_super = cv.warpPerspective(Img1.img, homo_mat_,stitch_size,borderValue=(0,0,0))
        img_transform = cv.warpPerspective(Img2.img, homo_mat,stitch_size,borderValue=(0,0,0))

        # Combine the image on one super image
        high_y = np.min(posVerts[:,1])
        img_transform[high_y:high_y,:,:] = 0
        img_super[img_transform>0]=0

        img_stitch = img_transform + img_super

        plt.figure(3)
        plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.show()

         # load the matching images
        img1 = cv.imread("dataset/origin_images/lamp_18.JPG")
        img2 = img_stitch

        ROIs1 = cv.selectROIs("select the area", img1)
        ROIs2 = cv.selectROIs("select the area", img2)

        for i in range(len(ROIs1)):
                ROIs1[i,2] = ROIs1[i,0] + ROIs1[i,2]
                ROIs1[i,3] = ROIs1[i,1] + ROIs1[i,3]
                ROIs2[i,2] = ROIs2[i,0] + ROIs2[i,2]
                ROIs2[i,3] = ROIs2[i,1] + ROIs2[i,3]

        # Initialize the object
        Img1 = Image(img1)
        Img2 = Image(img2)

        # Extract the features from each images
        kps1, des1 = Img1.findFeatures('sift')
        kps2, des2 = Img2.findFeatures('sift')   

        # Extract the masks to filter the features into several clusters
        masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
        masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)

        kpsCluster1, desCluster1 = Img1.featureCluster(masks1)
        kpsCluster2, desCluster2 = Img2.featureCluster(masks2)

        # Match the features with corresponding clusters in each image
        matches = utils.clusterMatch(desCluster1,desCluster2)


        # Integrate the clusters into one list
        kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)

        '''
        Find the parameters for homography matrix
        '''
        # Calculate the homography matrix for image transformation
        homo_mat, inliers_mask = utils.findHomography(matches, kps1_filter, kps2_filter)
        matches_inliers = list(itertools.compress(matches, inliers_mask))
        img_inliers = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches_inliers,None,**draw_params)


        print("\nNumber of inlier matches: ", len(matches_inliers),"\n")


        plt.figure(2)

        plt.imshow(cv.cvtColor(img_inliers, cv.COLOR_BGR2RGB))
        plt.title("Inlier Matches for Total Selected Area")
        plt.axis('off')


        # # Check the specific inliers
        # inlierCheck = input('Please Input [yes] to start checking the inliers:')
        # if inlierCheck == "yes":
        #     inliernum = int(input('Please Input the inlier Number you want to check:'))
        #     utils.inlierChecker(Img1,kps1_filter,Img2,kps2_filter,matches_inliers,draw_params,num=inliernum)
        

        '''
        Stitch the Images
        '''
        # Get the position of vertices
        posVerts = utils.transformVerts(img_size=np.array([Img1.img.shape[1],Img1.img.shape[0]]), homo_mat=homo_mat)
        # print("Left Top: ",posVerts[0,:],"\n",
        #       "Right Top: ",posVerts[1,:],"\n",
        #       "Right Bottom: ",posVerts[2,:],"\n",
        #       "Left Bottom: ",posVerts[3,:],"\n")
        
        x_min = posVerts[:,0].min()
        x_max = posVerts[:,0].max()
        y_min = posVerts[:,1].min()
        y_max = posVerts[:,1].max()
        print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %(x_min,x_max,y_min,y_max))

        stitch_size = (x_max+600,y_max+600)

        homo_mat_ = np.eye(3)
        img_super = cv.warpPerspective(Img1.img, homo_mat_,stitch_size,borderValue=(0,0,0))
        img_transform = cv.warpPerspective(Img2.img, homo_mat,stitch_size,borderValue=(0,0,0))

        # Combine the image on one super image
        high_y = np.min(posVerts[:,1])
        img_transform[high_y:high_y,:,:] = 0
        img_super[img_transform>0]=0

        img_stitch = img_transform + img_super

        plt.figure(3)
        plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.show()
        