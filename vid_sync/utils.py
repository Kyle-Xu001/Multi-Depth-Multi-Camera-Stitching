import os
import csv
import numpy as np
import glob
import json
import pandas as pd

def getTriggerNumbers(dirpath):
    lists_trigger_n = []
    
    meta_filelist = sorted(os.listdir(dirpath))
    for filename in meta_filelist:
        filepath = os.path.join(dirpath,filename)
        with open(filepath,'r') as f:
            reader = csv.reader(f,delimiter=',')
            list_trigger_n = []
            for row in reader:
                list_trigger_n.append(int(row[0]))
            lists_trigger_n.append(list_trigger_n)
            
    return lists_trigger_n

def getTriggerNumbers_pd(dirpath,_return_ros_epochs = False):
    lists_trigger_n = []
    lists_ros_epoch = []

    meta_filelist = sorted(os.listdir(dirpath))
    for filename in meta_filelist:
        filepath = os.path.join(dirpath,filename)
        with open(filepath, 'r') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.readline())
        df = pd.read_csv (filepath, header=None, sep=dialect.delimiter)
            
        lists_trigger_n.append(df[0].tolist())
        
        if _return_ros_epochs:
            
            lists_ros_epoch.append(df[4].tolist())
    
    if not _return_ros_epochs:
        return lists_trigger_n
    else:
        return lists_trigger_n,lists_ros_epoch

        


def findFrameIntervals(lists_trigger_n):
    first_trigger_n = [list_trigger_n[0] for list_trigger_n in lists_trigger_n]
    latest_first_trigger_n = max(first_trigger_n)
    while True:
        is_in_all_lists = True
        for list_trigger_n in lists_trigger_n:
            if not latest_first_trigger_n in list_trigger_n:
                is_in_all_lists = False
        if is_in_all_lists:
            break
        else:
            latest_first_trigger_n += 1
    
    last_trigger_n = [list_trigger_n[-1] for list_trigger_n in lists_trigger_n]
    earliest_last_trigger_n = min(last_trigger_n)
    while True:
        is_in_all_lists = True
        for list_trigger_n in lists_trigger_n:
            if not earliest_last_trigger_n in list_trigger_n:
                is_in_all_lists = False
        if is_in_all_lists:
            break
        else:
            earliest_last_trigger_n -= 1
    
    intervals_vid_pos = []
    for list_trigger_n in lists_trigger_n:
        ind_first_trigger_n = np.where(np.array(list_trigger_n)==latest_first_trigger_n)[0][0]
        ind_last_trigger_n = np.where(np.array(list_trigger_n)==earliest_last_trigger_n)[0][0]
        intervals_vid_pos.append((ind_first_trigger_n,ind_last_trigger_n))

    return intervals_vid_pos      

def getBB(json_dirpath):
    filepaths = sorted(glob.glob(json_dirpath+'/*json'))
    BB_lamps = []
    for filepath in filepaths:
        with open(filepath,'rb') as f:
            data = json.load(f)
            
        last_frame_id = data['images'][-1]['id']

        BB_frame = []
        BB_frames = []
        image_id = 0        
        annotations = iter(data['annotations'])
        annotation = next(annotations)
        while image_id <= last_frame_id:
            if image_id < annotation['image_id']:
                BB_frames.append(BB_lenToPts(BB_frame))
                BB_frame = []
                image_id += 1
            elif image_id == annotation['image_id']:
                BB_frame.append(annotation['bbox'])
                try:
                    annotation = next(annotations)
                except StopIteration:
                    BB_frames.append(BB_lenToPts(BB_frame))
                    n_pads = last_frame_id - image_id
                    padding = [np.array([])] * n_pads
                    BB_frames = BB_frames + padding
                    break
            else:
                raise RuntimeError('image_id somehow higher than image_id of annotation entry')

        BB_lamps.append(BB_frames)
        
    return BB_lamps

def BB_lenToPts(BBs):
    BBs = np.array(BBs)
    if BBs.size > 0:
        BBs[:,2] += BBs[:,0]
        BBs[:,3] += BBs[:,1]
    return BBs

def getTotalFrames(lists_frame_n,pos_frame_intervals):
    list_n_frames = []
    for list_frame_n,interval in zip(lists_frame_n,pos_frame_intervals):
        list_n_frames.append(list_frame_n[interval[1]] - list_frame_n[interval[0]] + 1)
    
    list_n_frames = [list_frame_n[interval[1]] - list_frame_n[interval[0]] + 1
                     for list_frame_n,interval in zip(lists_frame_n,pos_frame_intervals)]
    
    # Check the total number of frames in interval are the same for all videos
    for n_frames in list_n_frames:
        assert(n_frames == list_n_frames[0])
        
    return list_n_frames[0]

def rotate_BBs(BBs,img_size,angle):
    """
    Parameters
    ----------
    BBs : n x 4 ndarray
        ndarray of n bounding box boundaries, represented in each row as (xmin,ymin,xmax,ymax).
    img_size : iterable
        (width,height).
    angle : int
        Possible values: 90, 180, 270

    Raises
    ------
    ValueError
        When angle is invalid.

    Returns
    -------
    n x 4 ndarray
    """
    BBs_new = np.zeros(BBs.shape)
    if angle == 90:
        BBs_new[:,0] = BBs[:,1]
        BBs_new[:,1] = img_size[0] - BBs[:,2]
        BBs_new[:,2] = BBs[:,3]
        BBs_new[:,3] = img_size[0] - BBs[:,0]
    elif angle == 180:
        BBs_new[:,0] = img_size[0] - BBs[:,2]
        BBs_new[:,1] = img_size[1] - BBs[:,3]
        BBs_new[:,2] = img_size[0] - BBs[:,0]
        BBs_new[:,3] = img_size[1] - BBs[:,1]
    elif angle == 270:
        BBs_new[:,0] = img_size[1] - BBs[:,3]
        BBs_new[:,1] = BBs[:,0]
        BBs_new[:,2] = img_size[1] - BBs[:,1]
        BBs_new[:,3] = BBs[:,2]
    else:
        raise ValueError('Invalid rotation angle')
        
    return BBs_new
