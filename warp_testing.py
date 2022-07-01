import time
import cv2 as cv
import numpy as np
from stitch import remapStitch, simpleStitch
from stitch.utils import transformVerts

img1 = cv.imread("dataset/Arie/lamp_02_Arie.PNG")
img2 = cv.imread("dataset/Arie/lamp_01_Arie.PNG")

homo_mat = np.array([
        1.1244483061556527, 0.020214811271289677, 39.9989392156668,
        0.07199851056999242, 1.0588530742318294, 403.15979303614347,
        0.00010189396429253117, 0.00003398322398670004, 1.0]).reshape(3,3)

img_size = np.array([img2.shape[1],img2.shape[0]])
vertice = transformVerts(img_size, homo_mat)
x_min = vertice[:, 0].min()
x_max = vertice[:, 0].max()
y_min = vertice[:, 1].min()
y_max = vertice[:, 1].max()
# print("x_min: %d, x_max: %d y_min: %d, y_max: %d" %
#       (x_min, x_max, y_min, y_max))

# Define the size of the result image
x_max = np.max([x_max, img2.shape[1]])
y_max = np.max([y_max, img2.shape[0]])
stitch_size = (x_max, y_max)

img_transform = cv.warpPerspective(
        img2, homo_mat, stitch_size, borderValue=(0, 0, 0))
mask = img_transform>0
print(mask.shape)
print(mask)

T1 = time.time()
for i in range(100):
    img_stitch = simpleStitch(img1, img2, homo_mat,mask)
T2 = time.time()
cv.imshow("stitch",img_stitch)

x_range = np.arange(0, x_max)
y_range = np.arange(0, y_max)
u, v = np.meshgrid(x_range, y_range)
u = np.float32(u)
v = np.float32(v)

homo_mat = np.linalg.inv(homo_mat)
# warped_img = cv.remap(img1, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
# mask1 = np.ones((img1.shape[0],img1.shape[1]))
# warped_img = cv.remap(mask1, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
z_ = homo_mat[2,0]*u + homo_mat[2,1]*v + homo_mat[2,2]
map_x = (homo_mat[0,0]*u + homo_mat[0,1]*v + homo_mat[0,2])/z_
map_y = (homo_mat[1,0]*u + homo_mat[1,1]*v + homo_mat[1,2])/z_

T3 = time.time()
for i in range(100):
    img_stitch = remapStitch(img1,img2,u,v,map_x,map_y,mask)
T4 = time.time()
cv.imshow("warped",img_stitch)
cv.waitKey(0)
print("simpleStitch:",(T2-T1)*10)
print("remapStitch:",(T4-T3)*10)



