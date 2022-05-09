import itertools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import utils
from feature_extraction_test import Image
import image_match_test as match_utils

# Define the draw parameters for matching visualization
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (0,0,255),
                   flags = cv.DrawMatchesFlags_DEFAULT)

# load the matching images
img1 = cv.imread("dataset/origin_images/lamp_15_empty.JPG")
img2 = cv.imread("dataset/origin_images/lamp_14_empty.JPG")

#img1 = np.rot90(img1,1) 
#img2 = np.rot90(img2,1)

if (img1.shape[0]>img1.shape[1]):
# Manually define the ROI to locate the area for corresponding images
    # ROIs1 = [
    #     [450, 950, 768, 1300],
    #     [450, 750, 768, 1000],
    #     [450, 450, 768, 750],
    #     [450, 250, 768, 450]]
    # ROIs2 = [
    #     [0, 950,350, 1300],
    #     [0, 750, 350, 1000],
    #     [0, 500, 350, 800],
    #     [0, 250, 350, 500]]
    ROIs1 = [
        #[450, 850, 768, 1300],
        [475, 752, 767, 932],
        #[450, 300, 768, 625],
        [475, 260, 767, 422]]
    ROIs2 = [
        #[0, 900,350, 1300],
        [0, 752, 300, 942],
        #[0, 450, 350, 750],
        [0, 297, 300, 442]]
else:
    ROIs1 = [
        [180, 450, 500, 767],
        [600, 450, 900, 767]]

    ROIs2 = [
        [200, 0, 520, 350],
        [590, 0, 940, 350]]


# Initialize the object
Img1 = Image(img1)
Img2 = Image(img2)

# Extract the features from each images
kps1, des1 = Img1.kps, Img1.des
kps2, des2 = Img2.kps, Img2.des 

# Extract the masks to filter the features into several clusters
masks1 = utils.getMaskPointsInROIs(kps1, ROIs1)
masks2 = utils.getMaskPointsInROIs(kps2, ROIs2)

Img1.featureCluster(masks1)
Img2.featureCluster(masks2)

kpsCluster1, desCluster1 = Img1.kpsCluster, Img1.desCluster
kpsCluster2, desCluster2 = Img2.kpsCluster, Img2.desCluster

# Match the features with corresponding clusters in each image
matches = utils.clusterMatch(desCluster1,desCluster2)

# Show the number of matches
matchNum = 0
for i in range(len(matches)):
    matchNum += len(matches[i])
    print("-- Number of original matches in each area", len(matches[i]))
print("Number of original total matches: ", matchNum)

# draw the matches in each cluster
utils.drawMatch(Img1,kpsCluster1,Img2,kpsCluster2,matches,draw_params)


# Integrate the clusters into one list
kps1_filter, kps2_filter, matches =utils.featureIntegrate(kpsCluster1,kpsCluster2,matches)

# Visualize the total matches
plt.figure(1)
img_match = cv.drawMatches(Img1.img,kps1_filter,Img2.img,kps2_filter,matches,None,**draw_params)
plt.axis('off')
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.title("Feature Matching for Total Selected Area")



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



'''
Print the parameters of homography matrix
'''
np.set_printoptions(suppress=True)
print(homo_mat.flatten().tolist())


    


