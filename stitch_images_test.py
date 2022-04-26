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
    
    ap.add_argument("-hpp", "--homo_params_path", required=True, help="path to the stitching params") 
    
    ap.add_argument("-spp", "--stitch_params_path", required=True, help="path to the stitching order") 
    
    ap.add_argument("--farm_name", type=str, default="")
    
    ap.add_argument("--save_path", type=str, default="")
    
    args = vars(ap.parse_args())

    return args 


# Define the image stitching method for every frame
def stitchImages(imgs, homo_params, stitch_params):

    for image in stitch_params:
        for item in stitch_params[image]:
            # Define the image combination type
            stitch_type = stitch_params[image][item]["type"]
            
            if stitch_type == 'stitch':
                if stitch_params[image][item]["member"][0] != "img_stitch":
                    img1 = imgs[stitch_params[image][item]["member"][0]]
                else:
                    img1 = img_stitch
                    
                if stitch_params[image][item]["member"][1] != "img_stitch":
                    img2 = imgs[stitch_params[image][item]["member"][1]]
                else:
                    img2 = img_stitch

                img_stitch = ImageStitch.simpleStitch(img1, img2, homo_params[item])
            
            elif stitch_type == 'warp':
                if stitch_params[image][item]["member"][0] != "img_stitch":
                    img = imgs[stitch_params[image][item]["member"][0]]
                else:
                    img = img_stitch
                
                img_size = (img.shape[1], img.shape[0])
                img_stitch = cv2.warpPerspective(img, homo_params[item], img_size)
                
            elif stitch_type == 'rotate':
                if stitch_params[image][item]["member"][0] != "img_stitch":
                    img = imgs[stitch_params[image][item]["member"][0]]
                else:
                    img = img_stitch
                
                img_stitch = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            elif stitch_type == 'resize':
                if stitch_params[image][item]["member"][0] != "img_stitch":
                    img = imgs[stitch_params[image][item]["member"][0]]
                else:
                    img = img_stitch
                
                img_size = (stitch_params[image][item]["value"][0], stitch_params[image][item]["value"][1])
                img_stitch = cv2.resize(img, img_size)
                
            elif stitch_type == 'output':
                panorama = img_stitch
            
            
                # result_pos = stitch_params[image][item]["value"][0]
                # stitch_pos = stitch_params[image][item]["value"][1]
                # print(panorama.shape)
                # panorama[result_pos[0]:result_pos[1], result_pos[2]:result_pos[3],:] = img_stitch[stitch_pos[0]:stitch_pos[1], stitch_pos[2]:stitch_pos[3],:]
           
        
    return panorama


def stitch_all_frames(args):
    # Define the path to the video group (from lamp14-lamp23)
    vid_path = args["input_vid_group_path"]
    
    # Load the homography transform parameters
    homo_params = getParams(args["homo_params_path"])
    for homo_param in homo_params:
        # Transform the parameters from list to matrix
        homo_params[homo_param] = np.array(homo_params[homo_param]).reshape(-1, 3)
    
    # Load the stitch order params for stitching
    stitch_params = getParams(args["stitch_params_path"])

    panorama = np.zeros((stitch_params["img_size"][0],stitch_params["img_size"][1],3),dtype = np.uint8)
    
    stitch_params = stitch_params[args["farm_name"]]
    
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

    panorama = stitchImages(imgs, homo_params, stitch_params, panorama)
    panorama = cv2.rotate(panorama,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    width = panorama.shape[1]
    height = panorama.shape[0]
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
                    
        panorama = stitchImages(imgs, homo_params, stitch_params,panorama)
        panorama = cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        cv2.imshow(window_name,panorama)
        
        if save:
            writer.write(panorama)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return
        
    
    
if __name__ =='__main__':
    # Load the arguments
    args = parse_args()
    
    # Stitch the real-time video
    stitch_all_frames(args)
    
    # Destory the windows for real time
    cv2.destroyAllWindows()
