from . import utils
import cv2 as cv
import bisect
import os
import numpy as np

class Stream(object):
    def __init__(self,vid,list_trigger_n,interval_vid_pos,stream_id,list_ros_epoch = [],BB_frames=None):
        self.vid = vid
        self.vid.set(cv.CAP_PROP_POS_FRAMES,interval_vid_pos[0])
        
        self.list_trigger_n = list_trigger_n
        self.last_vid_pos = interval_vid_pos[1]
        
        self.img_prev = None
        self.stream_id = stream_id
        self.BB_frames = BB_frames
        self.list_ros_epoch = list_ros_epoch
        self.last_ros_epoch = self.list_ros_epoch[self.last_vid_pos]
    def getFrame(self,query_trigger_n):
        vid_pos = int(self.vid.get(cv.CAP_PROP_POS_FRAMES))
        
        if vid_pos > self.last_vid_pos:
            return None,[],self.last_ros_epoch
        
        if self.list_trigger_n[vid_pos] == query_trigger_n:
            success,img = self.vid.read()
            if not success:
                raise Exception('Failed to read from video')
            
            self.img_prev = img
        else:
            img = self.img_prev
        
        if self.BB_frames is not None:
            BB_frame = self.BB_frames[vid_pos]
            return img,BB_frame,self.list_ros_epoch[vid_pos]
        else:
            return img,[],self.list_ros_epoch[vid_pos]

        
    def skipToTriggerNum(self,trigger_n):
        # Find video position corresponding to trigger number
        ind_trigger_n = bisect.bisect_left(self.list_trigger_n,trigger_n)
        
        if ind_trigger_n == len(self.list_trigger_n):
            raise VideoEnd
            
        if self.list_trigger_n[ind_trigger_n] == trigger_n:
            self.vid.set(cv.CAP_PROP_POS_FRAMES,ind_trigger_n)
        else:
            self.vid.set(cv.CAP_PROP_POS_FRAMES,ind_trigger_n-1)
            success,self.img_prev = self.vid.read()
            if not success:
                raise Exception('Failed to read from video after skipping'
                                ' to trigger number '+trigger_n)
        
    def releaseVideo(self):
        self.vid.release()
        
    def getID(self):
        return self.stream_id
        
    def getVidPos(self):
        return int(self.vid.get(cv.CAP_PROP_POS_FRAMES))
    
    def getVidRes(self):
        return (int(self.vid.get(cv.CAP_PROP_FRAME_WIDTH)),
                int(self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)))

class VideoEnd(Exception):
    pass

class VideoSynchronizer(object):
    def __init__(self,dirpath,skip_init_frames=0,output_frame_info=False,use_distorted=False,get_BB=False):
        # Get videos
        if not use_distorted:
            dirpath_vids = os.path.join(dirpath,'videos')
            print(dirpath_vids)
        else:
            dirpath_vids = os.path.join(dirpath,'original_distorted')
        vid_name_list = sorted(os.listdir(dirpath_vids))
        vid_list = [cv.VideoCapture(os.path.join(dirpath_vids,vid_name)) for vid_name in vid_name_list]
        
        lists_trigger_n,lists_ros_epoch = utils.getTriggerNumbers_pd(os.path.join(dirpath,'meta'),_return_ros_epochs= True)
        intervals_vid_pos = utils.findFrameIntervals(lists_trigger_n)
        assert(len(vid_list)==len(lists_trigger_n) and len(vid_list)==len(intervals_vid_pos))
        
        # Get bounding boxes
        self.get_BB = get_BB
        self.stream_list = []
        if get_BB:
            dirpath_BB = os.path.join(dirpath,'bounding_box')
            BB_vids = utils.getBB(dirpath_BB)
            for (vid,list_trigger_n,interval_vid_pos,vid_name,BB_vid,list_ros_epoch) \
            in zip(vid_list,lists_trigger_n,intervals_vid_pos,vid_name_list,BB_vids,lists_ros_epoch):
                self.stream_list.append(Stream(vid,list_trigger_n,interval_vid_pos,vid_name[:6],list_ros_epoch = list_ros_epoch,BB_frames= BB_vid))
        else:
            for (vid,list_trigger_n,interval_vid_pos,vid_name,list_ros_epoch) \
            in zip(vid_list,lists_trigger_n,intervals_vid_pos,vid_name_list,lists_ros_epoch):
                self.stream_list.append(Stream(vid,list_trigger_n,interval_vid_pos,vid_name[:6],list_ros_epoch = list_ros_epoch))
        
        # Other variables
        self.n_frames = utils.getTotalFrames(lists_trigger_n,intervals_vid_pos)
        self.output_frame_info = output_frame_info
        self.current_trig_n = lists_trigger_n[0][intervals_vid_pos[0][0]]
        self.current_ros_epoch = lists_ros_epoch[0][intervals_vid_pos[0][0]]
        
        if skip_init_frames > 0:
            self.current_trig_n += skip_init_frames
            for stream in self.stream_list:
                stream.skipToTriggerNum(self.current_trig_n)
        
        self.current_trig_n -= 1
          
    def getFrames(self):
        self.current_trig_n += 1
        if self.output_frame_info:
            print('Trigger number:',self.current_trig_n)
        
        imgs = dict()
        BBs_frames = {}
        for stream in self.stream_list:
            img,BB_frame,self.current_ros_epoch = stream.getFrame(self.current_trig_n)
            imgs[stream.getID()] = img
            BBs_frames[stream.getID()] = BB_frame
        self.checkVideoEnd(imgs)
        
        return FrameSet(imgs,self.current_trig_n,BBs_frames,self.current_ros_epoch)
    
    # self.current_trig_n should be 1 less than frames because getFrames will advance 1 frame
    # Stream.skipToTriggerNum should receive trigger number 1 more than self.current_trig_n
    # because getFrames will advance 1 frame
    def skip(self,frames):
        if frames > 0:
            self.current_trig_n += frames-1
            for stream in self.stream_list:
                stream.skipToTriggerNum(self.current_trig_n+1)
        elif frames < 0:
            raise ValueError('Cannot skip negative number of frames')
    
    def checkVideoEnd(self,imgs):
        if any(img is None for img in imgs.values()):
            for img in imgs.values():
                assert(img is None)
            raise VideoEnd
    
    def close(self):
        for stream in self.stream_list:
            stream.releaseVideo()
    
    def getNStreams(self):
        return len(self.stream_list)
    
    def getTotalFrames(self):
        return self.n_frames
    
    def getTriggerNum(self):
        return self.current_trig_n
    
    def getStreamIDs(self):
        return [stream.getID() for stream in self.stream_list]
    
    def getVidRes(self,ID_query):
        for stream in self.stream_list:
            if stream.getID() == ID_query:
                return stream.getVidRes()
            
class FrameSet():
    def __init__(self,imgs,trig_n,BBs_frames,ros_epoch = ""):
        self.imgs = imgs
        self.trig_n = trig_n
        self.BBs_frames = BBs_frames
        self.ros_epoch_num = ros_epoch
        
    def resize(self,ID,scale):
        self.imgs[ID] = cv.resize(self.imgs[ID],(0,0),fx=scale,fy=scale)
        if len(self.BBs_frames[ID]) != 0:
            self.BBs_frames[ID] = np.around(self.BBs_frames[ID] * scale).astype(int)
        
    def rotate(self,ID,angle):
        if angle == 90:
            self.imgs[ID] = cv.rotate(self.imgs[ID],cv.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            self.imgs[ID] = cv.rotate(self.imgs[ID],cv.ROTATE_180)
        elif angle == 270:
            self.imgs[ID] = cv.rotate(self.imgs[ID],cv.ROTATE_90_CLOCKWISE)
        else:
            raise ValueError('Invalid rotation flag')
            
        img_size = self.imgs[ID].shape[1::-1]
        if len(self.BBs_frames[ID]) != 0:
            self.BBs_frames[ID] = utils.rotate_BBs(self.BBs_frames[ID],img_size,angle)