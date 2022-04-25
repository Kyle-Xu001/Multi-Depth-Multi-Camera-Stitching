"""
Created on Thu Apr 7 10:41:05 2022

@author: Chenghao Xu

This is the test of new function `stitchImages` based on the stitch_custom.py
"""

import cv2
import os
import argparse
import ImageStitch
import numpy as np
from utils import getParams
from pathlib import Path
from vid_sync import VideoSynchronizer
import matplotlib.pyplot as plt

# Create visualize window for Videos
def create_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    cv2.resizeWindow(window_name, 800,600) # Decrease the window size


# Define the function to Load the arguments
def parse_args():
    ap = argparse.ArgumentParser(description='Run multi-camera stitching')

    ap.add_argument("-ivid", "--input_vid_group_path", required=True, help="path to the input video group")
    
    ap.add_argument("-sp", "--stitch_params_path", required=True, help="path to the stitching params") 
    
    ap.add_argument("-so", "--stitch_order_path", required=True, help="path to the stitching order") 
    
    ap.add_argument("--farm_name", type=str, default="")
    
    ap.add_argument("--save_path", type=str, default="")
    
    args = vars(ap.parse_args())

    return args 


# Define the image stitching method for every frame
def stitchImages(imgs, homo_params, stitch_order):
    '''TO DO: Try to simplify this function'''
    for image in stitch_order:
        for item in stitch_order[image]:
            # Define the image combination type
            stitch_type = stitch_order[image][item]["type"]
            
            if stitch_type == 'stitch':
                if stitch_order[image][item]["member"][0] != "img_stitch":
                    img1 = imgs[stitch_order[image][item]["member"][0]]
                else:
                    img1 = img_stitch
                    
                if stitch_order[image][item]["member"][1] != "img_stitch":
                    img2 = imgs[stitch_order[image][item]["member"][1]]
                else:
                    img2 = img_stitch

                img_stitch = ImageStitch.simpleStitch(img1, img2, homo_params[item])
            
            elif stitch_type == 'warp':
                if stitch_order[image][item]["member"][0] != "img_stitch":
                    img = imgs[stitch_order[image][item]["member"][0]]
                else:
                    img = img_stitch
                
                img_size = (img.shape[1], img.shape[0])
                img_stitch = cv2.warpPerspective(img, homo_params[item], img_size)
        
        #panorama = 
            plt.figure(0)
            plt.imshow(img_stitch)
            plt.show()
            
    if farm_name == 'Mathe':
        # Stitch the right area of Mathe farm
        img_stitch_right = ImageStitch.simpleStitch(imgs["lamp02"], imgs["lamp03"], homo_params["lamp03-lamp02"])
        img_stitch_right = ImageStitch.simpleStitch(img_stitch_right, imgs["lamp04"], homo_params["lamp04-lamp03"])
        img_stitch_right = ImageStitch.simpleStitch(img_stitch_right, imgs["lamp05"], homo_params["lamp05-lamp04"])
        img_stitch_right = ImageStitch.simpleStitch(img_stitch_right, imgs["lamp06"], homo_params["lamp06-lamp05"])
        
        # Correction of right stitching image
        img_stitch_right = cv2.warpPerspective(img_stitch_right, homo_params["lamp02-lamp06"], (img_stitch_right.shape[1], img_stitch_right.shape[0]))
        img_stitch_right = cv2.warpPerspective(img_stitch_right, homo_params["lamp02-lamp06-shrink"], (img_stitch_right.shape[1], img_stitch_right.shape[0]))
    
        # Resize the corridor image (lamp07)
        img_corridor = cv2.warpPerspective(imgs["lamp07"], homo_params["lamp07-correction"], (imgs["lamp07"].shape[1], imgs["lamp07"].shape[0]))
        img_corridor = cv2.rotate(img_corridor,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_corridor = cv2.resize(img_corridor, (450, 900))
        
        # Stitch the middle area of Mathe farm
        img_stitch_middle = ImageStitch.simpleStitch(imgs["lamp15"], imgs["lamp14"], homo_params["lamp15-lamp14"])
        img_stitch_middle = ImageStitch.simpleStitch(imgs["lamp16"], img_stitch_middle, homo_params["lamp16-lamp15"])
        img_stitch_middle = ImageStitch.simpleStitch(imgs["lamp17"], img_stitch_middle, homo_params["lamp17-lamp16"])
        img_stitch_middle = ImageStitch.simpleStitch(imgs["lamp18"], img_stitch_middle, homo_params["lamp18-lamp17"])
        img_stitch_middle = cv2.warpPerspective(img_stitch_middle, homo_params["Right_Transform"], (img_stitch_middle.shape[1], img_stitch_middle.shape[0]))
        
        # Stitch the left area of Mathe farm
        img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp22"], 0), cv2.flip(imgs["lamp23"], 0), homo_params["lamp23-lamp22"])
        img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp21"], 0), img_stitch_left, homo_params["lamp22-lamp21"])
        img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp20"], 0), img_stitch_left, homo_params["lamp21-lamp20"])
        img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp19"], 0), img_stitch_left, homo_params["lamp20-lamp19"])
        img_stitch_left = cv2.warpPerspective(img_stitch_left, homo_params["Left_Transform"], (img_stitch_left.shape[1], img_stitch_left.shape[0]))
        img_stitch_left = cv2.flip(img_stitch_left, 0)
        
        # Stitch the left area and middle area
        img_stitch = ImageStitch.simpleStitch(img_stitch_left, img_stitch_middle, homo_params["stitch_total"])
        
        # Create a empty image and combine all stitching images
        panorama = np.zeros((7650,1300,3),dtype = np.uint8)
        panorama[4500:5400, 825:1275,:] = img_corridor
        panorama[0:4600,:,:] = img_stitch[600:5200,100:1400,:]
        panorama[5150:7550,:,:] = img_stitch_right[100:2500,20:1320,:]
        
    elif farm_name == 'Arie':
        img_stitch = ImageStitch.simpleStitch(imgs["lamp02"], imgs["lamp01"], homo_params["lamp02-lamp01"])
        img_stitch = ImageStitch.simpleStitch(imgs["lamp03"], img_stitch, homo_params["lamp03-lamp02"])
        img_stitch = ImageStitch.simpleStitch(imgs["lamp04"], img_stitch, homo_params["lamp04-lamp03"])
        img_stitch = ImageStitch.simpleStitch(imgs["lamp05"], img_stitch, homo_params["lamp05-lamp04"])
        panorama = img_stitch
        
    elif farm_name == 'office_farm':
        img_stitch = ImageStitch.simpleStitch(imgs["lamp02"], imgs["lamp03"], homo_params["lamp03-lamp02"])
        img_stitch = ImageStitch.simpleStitch(imgs["lamp01"], img_stitch, homo_params["lamp02-lamp01"])
        panorama = img_stitch
        
    return panorama


def stitch_all_frames(args):
    # Define the path to the video group (from lamp14-lamp23)
    vid_path = args["input_vid_group_path"]
    
    # Load the homography transform parameters
    homo_params = getParams(args["stitch_params_path"])
    for homo_param in homo_params:
        # Transform the parameters from list to matrix
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    # Load the stitch order params for stitching
    stitch_order = getParams(args["stitch_order_path"])[args["farm_name"]]

    
    # Define the function about saving the stitching video
    save = False
    if args["save_path"] == True:
        save = True
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        p = Path(os.path.join(args["save_path"],"stitched"))
        p.mkdir(parents=True,exist_ok=True)
        
        save_video_path = os.path.join(args["save_path"],"stitched",vid_path.split('/')[-1]+"_stitch.mp4")
    
    
    # Initialize the stitching to estimate the parameters
    vid_sync_test = VideoSynchronizer(vid_path, use_distorted=False, skip_init_frames=0)
    frameset = vid_sync_test.getFrames()
    imgs = frameset.imgs

    img_stitch = stitchImages(imgs, homo_params, stitch_order)
    img_stitch = cv2.rotate(img_stitch,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    width = img_stitch.shape[1]
    height = img_stitch.shape[0]
    if save:
        writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
    
    # Create OpenCV windows for each videos    
    window_name = "stitched_image"
    create_window(window_name)
    
    # for img_src in imgs.keys():
    #     create_window(img_src)

    # Start Real-Time Stitching
    vid_sync = VideoSynchronizer(vid_path,use_distorted=False,skip_init_frames=0)
    
    while(True):
        frameset = vid_sync.getFrames()
        imgs = frameset.imgs
                    
        img_stitch = stitchImages(imgs, homo_params, stitch_order)
        img_stitch = cv2.rotate(img_stitch,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.imshow(window_name,img_stitch)
        
        if save:
            writer.write(img_stitch)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
    
    
if __name__ =='__main__':
    # Load the arguments
    args = parse_args()
    
    # Stitch the real-time video
    stitch_all_frames(args)
    
    # Destory the windows for real time
    cv2.destroyAllWindows()
