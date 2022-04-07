"""
Created on Thu Apr 7 10:41:05 2022

@author: Chenghao Xu
"""
import cv2
import os
import argparse
import ImageStitch
import numpy as np
from utils import getHomoParams
from pathlib import Path
from vid_sync import VideoSynchronizer

def create_window(window_name):
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    cv2.resizeWindow(window_name, 800,600)
    
def parse_args():
    ap = argparse.ArgumentParser(description='Run multi-camera stitching')

    ap.add_argument("-ivid", "--input_vid_group_path", required=True, help="path to the input video group")
    
    ap.add_argument("-pst", "--stitch_params_path", required=True, help="path to the stitching params") 
    
    ap.add_argument("--save_path", type=str, default="")
    
    args = vars(ap.parse_args())

    return args 
def stitchImages(imgs, homo_params):
    img_stitch_right = ImageStitch.simpleStitch(imgs["lamp15"], imgs["lamp14"], np.array(homo_params["lamp15-lamp14"]).reshape(-1, 3))
    img_stitch_right = ImageStitch.simpleStitch(imgs["lamp16"], img_stitch_right, np.array(homo_params["lamp16-lamp15"]).reshape(-1, 3))
    img_stitch_right = ImageStitch.simpleStitch(imgs["lamp17"], img_stitch_right, np.array(homo_params["lamp17-lamp16"]).reshape(-1, 3))
    img_stitch_right = ImageStitch.simpleStitch(imgs["lamp18"], img_stitch_right, np.array(homo_params["lamp18-lamp17"]).reshape(-1, 3))
    img_stitch_right = cv2.warpPerspective(img_stitch_right, np.array(homo_params["Right_Transform"]).reshape(-1, 3), (img_stitch_right.shape[1], img_stitch_right.shape[0]))
    
    img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp22"], 0), cv2.flip(imgs["lamp23"], 0), np.array(homo_params["lamp23-lamp22"]).reshape(-1, 3))
    img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp21"], 0), img_stitch_left, np.array(homo_params["lamp22-lamp21"]).reshape(-1, 3))
    img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp20"], 0), img_stitch_left, np.array(homo_params["lamp21-lamp20"]).reshape(-1, 3))
    img_stitch_left = ImageStitch.simpleStitch(cv2.flip(imgs["lamp19"], 0), img_stitch_left, np.array(homo_params["lamp20-lamp19"]).reshape(-1, 3))
    img_stitch_left = cv2.flip(img_stitch_left, 0)
    img_stitch_left = cv2.warpPerspective(img_stitch_left, np.array(homo_params["Left_Transform"]).reshape(-1, 3), (img_stitch_left.shape[1], img_stitch_left.shape[0]))
    
    img_stitch = ImageStitch.simpleStitch(img_stitch_left, img_stitch_right, np.array(homo_params["stitch_total"]).reshape(-1, 3))
    img_stitch = img_stitch[500:5300,50:1450,:]
    return img_stitch


def stitch_all_frames(args):
    vid_path = args["input_vid_group_path"]
    
    save = False
    if args["save_path"] == True:
        save = True
        fps = 20.0
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        p = Path(os.path.join(args["save_path"],"stitched"))
        p.mkdir(parents=True,exist_ok=True)
        
        save_video_path = os.path.join(args["save_path"],"stitched",vid_path.split('/')[-1]+"_stitch.mp4")

    homo_params = getHomoParams(args["stitch_params_path"])
    stream_ids = sorted(list(homo_params.keys()))

    
    vid_sync_test = VideoSynchronizer(vid_path, use_distorted=False, skip_init_frames=0)
    frameset = vid_sync_test.getFrames()
    imgs = frameset.imgs

    img_stitch = stitchImages(imgs, homo_params)
    img_stitch = cv2.rotate(img_stitch,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    width = img_stitch.shape[1]
    height = img_stitch.shape[0]
    if save:
        writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
        
    window_name = "stitched_image"
    create_window(window_name)
    
    for img_src in imgs.keys():
        create_window(img_src)

    vid_sync = VideoSynchronizer(vid_path,use_distorted=False,skip_init_frames=0)
    
    while(True):
        frameset = vid_sync.getFrames()
        imgs = frameset.imgs
        for img_src, img_arr in imgs.items():
            cv2.imshow(img_src,img_arr)
            
        img_stitch = stitchImages(imgs,homo_params)
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

    