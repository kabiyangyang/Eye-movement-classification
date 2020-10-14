
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 12:25:00 2020

@author: ThinkPad
"""


import numpy as np
import os
import cv2
import h5py
from scipy import ndimage
from argparse import ArgumentParser

TAG_FLOAT = 202021.25

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def parse_arguments():

    parser = ArgumentParser('masks propagation and delete still object masks')


    parser.add_argument('--flow-path', default='F:/of2/',
                        help='The path that contains the flow data')
    
    parser.add_argument('--video-path', default='E:/movies-m2t/',
                        help='The path that contains the flow data')

    parser.add_argument('--output-path', default='D:/Masks/propagated_masks/',
                        help='Folder containing the files with already extracted features.')

    parser.add_argument('--outputvideo-path', default='D:/Masks/videos/',
                        help='Folder containing the files with already extracted features.')
    
    
    return parser.parse_args()


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def read(file):

	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))	
	f.close()

	return flow


def backgroundsubstractor(vidlist):
    MOG_mask = []
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for frame in vidlist:
        fgmask = fgbg.apply(frame, learningRate = 0.01)
        #res = np.zeros(fgmask.shape)            
        fgmask[fgmask!=0] = 255 
        th = cv2.medianBlur(fgmask.copy(), 5)
        
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 1)
        MOG_mask.append(dilated)
    return MOG_mask

def add_new_mask(mog_mask, propagated_nxt, res):
    tmp_res = mog_mask * propagated_nxt
    vals1, counts1 = np.unique(tmp_res, return_counts = True)
    vals2, counts2 = np.unique(propagated_nxt,return_counts = True)
    max_val = np.max(res)
    if(vals1.shape[0]>1):
        for val in vals1[1:]:
            tmp_frame = np.zeros((mog_mask.shape))
            count1 = counts1[vals1==val]
            count2 = counts2[vals2==val]
            if(count1>10 and count1>count2/5):
               tmp_idxs = np.argwhere(propagated_nxt == val)
               tmp_vals = res[tmp_idxs.T[0], tmp_idxs.T[1]]
               
               val3 = np.unique(tmp_vals)
               if(val3[0] == 0 and val3.shape[0]==1):
                   tmp_frame[tmp_res==val] += int(max_val+val)
                   dilated = cv2.dilate(tmp_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations = 1)
                   res[dilated!=0] =int(max_val+val)



def forward_propagation(mot_mask, flow_path, mog_mask):
    flownames = os.listdir(flow_path)
    res = mot_mask.copy()

    for j in np.arange(1,  min(len(mot_mask), len(flownames))-1):
        flow = read(flow_path + flownames[j])
        tmp_frame = np.zeros(mot_mask[j].shape)
        res[j] = ndimage.median_filter(res[j], 10)
        idxs = np.argwhere(res[j]!=0)

        vals = res[j][idxs.T[0], idxs.T[1]]
      
        nxt_frame_idx = np.round(idxs + flow[idxs.T[0], idxs.T[1]]).astype('int')
        vals = vals[(nxt_frame_idx[:,0]<res[j].shape[0]) & (nxt_frame_idx[:,1]<res[j].shape[1])]
        nxt_frame_idx = nxt_frame_idx[nxt_frame_idx[:,0]<res[j].shape[0]]
        nxt_frame_idx = nxt_frame_idx[nxt_frame_idx[:,1]<res[j].shape[1]]

        propagated_nxt = np.zeros(res[j].shape)
        propagated_nxt[nxt_frame_idx.T[0], nxt_frame_idx.T[1]] = vals
        add_new_mask(mog_mask[j+1], propagated_nxt, res[j+1])
    return res
        

                
def backward_propagation(tmp_mask, flow_path, mog_mask):
    flownames = os.listdir(flow_path)
    mot_mask= tmp_mask
    res = mot_mask.copy()
    #reverse order
    for j in np.arange(1,  min(len(mot_mask), len(flownames))-1)[::-1]:
        flow = read(flow_path + flownames[j])
        #vals = np.unique(res[j])
        tmp_frame = np.zeros(mot_mask[j].shape)
        res[j] = ndimage.median_filter(res[j], 10)
        idxs = np.argwhere(res[j]!=0)

        vals = res[j][idxs.T[0], idxs.T[1]]
      
        nxt_frame_idx = np.round(idxs - flow[idxs.T[0], idxs.T[1]]).astype('int')
        vals = vals[(nxt_frame_idx[:,0]<res[j].shape[0]) & (nxt_frame_idx[:,1]<res[j].shape[1])]
        nxt_frame_idx = nxt_frame_idx[nxt_frame_idx[:,0]<res[j].shape[0]]
        nxt_frame_idx = nxt_frame_idx[nxt_frame_idx[:,1]<res[j].shape[1]]

        propagated_nxt = np.zeros(res[j].shape)
        propagated_nxt[nxt_frame_idx.T[0], nxt_frame_idx.T[1]] = vals
        add_new_mask(mog_mask[j-1], propagated_nxt, res[j-1])
    return res

def run(args):        
    #['breite_strasse', 'bridge_1','bridge_2', 'doves', 'ducks_children', 'golf']#
    VIDLIST =  ['koenigstrasse']#['holsten_gate','puppies', 'roundabout', 'sea', 'st_petri_gate', 'street', 'st_petri_mcdonalds']
    for v in VIDLIST:
        flow_path = args.flow_path + v + '/'
        
        vid = args.video_path + v + '.m2t'
        PATH = args.video_path +v +  '/seg/final_mask_arr.h5'#'E:/movies-m2t/st_petri_mcdonalds/seg/final_mask_arr.h5'
       # PATH2 = 'D:/Masks/mog_masks/MOG_' + v + '.h5'
        OUTPUTPATH = args.output_path + 'MOT_' + v + '.h5' 
        OUTPUTVIDEOPATH = args.outputvideo_path +'MOT'+ v + '.avi'
        cap = cv2.VideoCapture(vid)
        vidlist = []
        mot_mask = []
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(OUTPUTVIDEOPATH , fourcc, 10, (1280,720))
        count = 0
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            vidlist.append(np.array(frame))
            count+=1
            if (count > (video_length-1)):
                cap.release()
                break
            
        with h5py.File(PATH, "r") as f:
             obj_mask = f['data'][:]
        
        mog_mask = backgroundsubstractor(vidlist)
        
        for i in np.arange(0, len(vidlist)):
            if(i<obj_mask.shape[0]):
                mog_mask[i][mog_mask[i]!=0]=1
                tmp_mask = mog_mask[i]*obj_mask[i]
                tmp = np.zeros((720, 1280), dtype = 'int')
                vals, counts = np.unique(tmp_mask, return_counts = True)
        
                if(vals.shape[0]>1):
                    for val in vals[1:]:
                        if(counts[vals==val][0]>5 and counts[vals==val][0]>len(np.where(obj_mask[i]==val)[0])/20):
        
                            tmp[obj_mask[i]==val] = val
        
                mot_mask.append(tmp)
        print('motion mask calculation finished')
        del obj_mask
        mot_mask2 = np.copy(mot_mask)
        res = forward_propagation(mot_mask2, flow_path, mog_mask)
        res = backward_propagation(res, flow_path, mog_mask)  
        del mog_mask
        print('propagation finished')
        with h5py.File(OUTPUTPATH, 'w') as hf:
            hf.create_dataset('data', data = np.array(res, dtype = 'int'))
        
        for i in np.arange(1, len(res)):      
               # vals, counts = np.unique(res[i], return_counts = True)
                
                image = vidlist[i]
                color = [1,0,0]
                alp = 0.3
                color2 = [0,0,1]
                diff = res[i]-mot_mask[i]
        
                  #if(vals.shape[0]>1)      
                for c in range(3):
                    image[:, :, c] = np.where(res[i]!=0,
                                                     image[:, :, c] *
                                                     (1 - alp) + alp * color[c] * 255,
                                                     image[:, :, c])
                    image[:,:,c] = np.where(diff!=0,
                                                  image[:, :, c] *
                                                  (1 - alp) + alp * color2[c] * 255,
                                                  image[:, :, c])
                cv2.putText(image, str(i), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, [255,0, 255], 2)
        
                video.write(image)
        video.release()
        del video


def __main__():
    args = parse_arguments()
    run(args)   
if __name__ == '__main__':
    __main__()  



    


    
    