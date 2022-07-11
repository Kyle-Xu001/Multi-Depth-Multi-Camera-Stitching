
import cv2
import numpy as np
import random
# 首先读入img
img = cv2.imread("dataset/Arie/lamp_02_Arie.PNG")
#img = cv2.resize(img,(180,32))
# N对基准控制点
N=100
points=[]
dx=int(180/(N-1))
for i in range(2*N):
    points.append((dx*i,4))
    points.append((dx*i,36))
# 周围拓宽一圈
#img = cv2.copyMakeBorder(img,4,4,0,0,cv2.BORDER_REPLICATE)
# 画上绿色的圆圈
# for point in points:
# 	cv2.circle(img, point, 1, (0, 255, 0), 2)
tps = cv2.createThinPlateSplineShapeTransformer()
 
sourceshape = np.array(points,np.int32)
sourceshape=sourceshape.reshape(1,-1,2)
matches =[]
for i in range(1,N+1):
    matches.append(cv2.DMatch(i,i,0))
 
# 开始随机变动
newpoints=[]
PADDINGSIZ=10
for i in range(N):
    nx=points[i][0]+random.randint(0,PADDINGSIZ)-PADDINGSIZ/2
    ny=points[i][1]+random.randint(0,PADDINGSIZ)-PADDINGSIZ/2
    newpoints.append((nx,ny))
print(points,newpoints)
targetshape = np.array(newpoints,np.int32)
targetshape=targetshape.reshape(1,-1,2)

tps.estimateTransformation(sourceshape,targetshape ,matches)
img=tps.warpImage(img)

cv2.imshow("warped",img)
cv2.waitKey(0)
print(sourceshape)
print(targetshape)