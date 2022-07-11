import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from stitch import utils, Stitch, simpleStitch

'''This script is used for stitching two undistortion images with basic workflow'''

# Define the draw parameters for matching visualization
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (0,0,255),
                   flags = cv.DrawMatchesFlags_DEFAULT)

# load the matching images
img1 = cv.imread("dataset/Mathe/lamp_02_Arie.PNG")
img2 = cv.imread("dataset/Mathe/lamp_01_Arie.PNG")
# img1 = cv.imread("dataset/example_image/APAP-railtracks/1.JPG")
# img2 = cv.imread("dataset/example_image/APAP-railtracks/2.JPG")
#img1 = cv.imread("dataset/example_image/NISwGSP-denny/denny02.jpg")
#img2 = cv.imread("dataset/example_image/NISwGSP-denny/denny03.jpg")

#img1 = np.rot90(img1,1) 
#img2 = np.rot90(img2,1)

# The stitch order need to be flipped to be keep the right image unchanged
#img1_ = cv.flip(img1_, 0)
#img2_ = cv.flip(img2_, 0)

'''Extract Features within Interested Regions'''
# Manually define the ROI to locate the area for corresponding images
cv.namedWindow("Area Selection", cv.WINDOW_NORMAL)
cv.resizeWindow("Area Selection", 800, 600)
ROIs1 = cv.selectROIs("Area Selection", img1)
ROIs2 = cv.selectROIs("Area Selection", img2)

for i in range(len(ROIs1)):
    ROIs1[i, 2] = ROIs1[i, 0] + ROIs1[i, 2]
    ROIs1[i, 3] = ROIs1[i, 1] + ROIs1[i, 3]
    ROIs2[i, 2] = ROIs2[i, 0] + ROIs2[i, 2]
    ROIs2[i, 3] = ROIs2[i, 1] + ROIs2[i, 3]


# Initialize the object
stitch = Stitch(img1, img2)
stitch.featureExtract(ROIs1, ROIs2)


'''Match the Corresponding Features'''
# Define the matches based on two images
matches_list = stitch.clusterMatch('sift', knn=True)

# Show the number of matches
matchNum = 0
for i in range(len(matches_list)):
    matchNum += len(matches_list[i])
    print("-- Number of original matches in each area", len(matches_list[i]))
print("Number of original total matches: ", matchNum)

# draw the matches in each cluster
utils.drawMatch(stitch.Img1,stitch.Img2,matches_list,draw_params)

# Combine the features in one lists from each cluster
stitch.featureIntegrate(matches_list)

# Visualize the total matches
plt.figure(1)
img_match = cv.drawMatches(img1,stitch.Img1.kps,img2,stitch.Img2.kps,stitch.matches,None,**draw_params)
plt.axis('off')
plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
plt.title("Feature Matching for Total Selected Area")


'''
Find the parameters for homography matrix
'''
# Calculate the homography matrix for image transformation
homo_mat, matches_inliers = stitch.homoEstimate()
img_inliers = cv.drawMatches(img1,stitch.Img1.kps,img2,stitch.Img2.kps,matches_inliers,None,**draw_params)


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
img_stitch = simpleStitch(img1, img2, homo_mat)

plt.figure(3)
plt.imshow(cv.cvtColor(img_stitch, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

'''Test the TPS transform method'''
Y = []
X = []
new_matches = []
for i in range(40):
# for i in range(5):
    Y.append(stitch.Img2.kps[matches_inliers[4*i].trainIdx])
    X.append(stitch.Img1.kps[matches_inliers[4*i].queryIdx])
    new_matches.append(cv.DMatch(i, i, 0))
Y = cv.KeyPoint_convert(Y)
X = cv.KeyPoint_convert(X)

Y = np.asarray(Y)
Y = Y.reshape([1,-1,2])
X = np.asarray(X)
X = X.reshape([1,-1,2])
Y = cv.perspectiveTransform(Y, homo_mat)

homo_mat_ = np.eye(3)
img_super = cv.warpPerspective(img1, homo_mat_, (img_stitch.shape[1],img_stitch.shape[0]))
img_transform = cv.warpPerspective(img2, homo_mat, (img_stitch.shape[1],img_stitch.shape[0]))

print(Y-X)
tps = cv.createThinPlateSplineShapeTransformer()
tps.estimateTransformation(Y, X, new_matches)
print(tps.applyTransformation(Y))
img_transform=tps.warpImage(img_transform)
img_super[img_transform > 0] = 0
img_transform= img_transform + img_super

plt.figure(4)
plt.imshow(cv.cvtColor(img_transform, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

'''
Print the parameters of homography matrix
'''
np.set_printoptions(suppress=True)
print(homo_mat.flatten().tolist())


    


