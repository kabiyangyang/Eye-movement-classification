# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:01:59 2020

@author: ThinkPad
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:59:35 2020

@author: ThinkPad
"""
from argparse import ArgumentParser
import arff
import os
import cv2
import numpy as np
import flow_calculation
import preproc
import matplotlib.pyplot as plt
import h5py



TAG_FLOAT = 202021.25
translate_dict = {'UNKNOWN' : 0, 'FIX':1,'SACCADE':2, 'SP':3,  "NOISE" :4}

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



def parse_arguments():

    parser = ArgumentParser('flow feature processing')

    parser.add_argument('--gt-path',  default= 'D:/deep_em_classifier-master/deep_em_classifier-master/GazeCom/ground_truth/',#'D:/deep_em_classifier-master/deep_em_classifier-master/GazeCom/ground_truth/',#D:/deep_em_classifier-master/deep_em_classifier-master/GazeCom/ground_truth/',
                        help='The path that contains the ground truth data with hand labeled classes')
    parser.add_argument('--flow-path', default= 'H:/of_data/',#'H:/of_data/',
                        help='The path that contains the flow data')
    parser.add_argument('--video-path', default='E:/movies-m2t/', #'E:/movies-m2t/',#'G:/Hollywood2-actions/test/',#'H:/EyeMovementDetectorEvaluation/EyeMovementDetectorEvaluation/Stimuli/videos/',#'D:/Masks/propagated_masks/',
                        help='Folder containing the video stimuli.')
    parser.add_argument('--mask-path', default= 'H:/gazeCom_mask/',#'H:/gazeCom_mask/', 
                        help='Folder containing the masks.')
    parser.add_argument('--output-path',  default='../data/hollywood2_features/',
                        help='Folder containing the output data.')
    
    parser.add_argument('--feature-scales', nargs='+', default=[8, 16, 24, 32], type=int,
                        help='temproal scales for the flow and gaze moving direction distances')
    
    parser.add_argument('--pre-processing', dest='pre_processing', action='store_true',
                        help='Preprocessing on the coordinats for feature extraction on video stimuli, the output coords remain the same.')
    
    parser.add_argument('--velocity-space', dest='velocity_space', action='store_true',
                        help='Preprocessing on the coordinats for feature extraction on video stimuli, the output coords remain the same.')
   
   
    return parser.parse_args()




def moving_average_2(a, n) :
    """
    Param a: to be averaged data
    Param n: moving average window size
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



def slidingwindow(arr_x, arr_y, fx, fy,  window_width_interval):
    """
    Param arr_x: gaze position in X
    Param arr_y: gaze position in Y
    Param fx: flow trajectory in x
    Param fy: flow trajectory in y
    Param window_width_interval: user defined temproal scales for direction difference (prefer a coarser scale than other features)
    and gaze moving direction distances
    """

    x = moving_average_2(arr_x, 9)
    y = moving_average_2(arr_y, 9)
    flow_x = moving_average_2(fx, 9)
    flow_y = moving_average_2(fy, 9)
    tmp1 = np.vstack((x, y)).T
    tmp2 = np.vstack((flow_x, flow_y)).T
    
    outputdata = np.hstack((tmp1, tmp2))
    
    for window_width in window_width_interval:
        step = window_width/2
        f_dir = []
        p_dir = []

        for i  in np.arange(0,x.shape[0]):

            
            if(step == window_width):
                startPos = i - step
                endPos = int(i)
            else:
                startPos = int(i-step)
                endPos = int(i+step)
                
            if(startPos<1):
                startPos = int(i)
            if(endPos>len(x)-1):
                endPos = int(i)
                
            if(startPos == endPos):
                f_dir.append(0)
                p_dir.append(0)
                continue
            
            tmp_f_dir = np.arctan2(np.sum(flow_y[startPos:endPos]), np.sum(flow_x[startPos:endPos]) )
            tmp_p_dir = np.arctan2((y[endPos]-y[startPos]), (x[endPos] - x[startPos]))
                      
            
            f_dir.append(tmp_f_dir)
            p_dir.append(tmp_p_dir)

        res = np.asarray(f_dir) - np.asarray(p_dir)
        
        d_res = np.degrees(res)
        
        d_res[d_res>180] = d_res[d_res>180] - 360
        d_res[d_res<-180] = d_res[d_res<-180] + 360
        
        
        outputdata = np.hstack((outputdata, d_res.reshape(-1,1)))

    return outputdata

def find_listoftuple(list_of_tuple, element):
    return [i for i, tupl in enumerate(list_of_tuple) if tupl[0] == element][0]


def run(args):
    """
    run function: for each video and each participants, find out the nearest object and apply
    object mask on the flow data, then preprocessing the data and save them as featrues into the .arff files.
    
    """
    
    fps = 30
    videos_run = os.listdir(args.video_path)
    for video in videos_run:
        
        # video name
        v = os.path.splitext(video)[0] 
        
        
        if(v[0] == '.'):
            continue
        
        videoloc = args.video_path + video
        outputfolder = args.output_path + v + '/'
        
        #if no output path, create
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
        #read in the frames  
        cap = cv2.VideoCapture(videoloc)
        vidlist = []
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        count = 0
        while cap.isOpened():
            # Extract the frame
            ret, frame = cap.read()
            vidlist.append(np.array(frame))
            count+=1
            if (count > (video_length-1)):
                cap.release()
                break         

        mask_path = args.mask_path + v + '/final_mask_arr.h5'
        of_path = args.flow_path + v  + '/'
        
        of_files = os.listdir(of_path)
        suffix = os.path.splitext(of_files[0])[1]
        
        #choose whether the optical flow data is saved in '*.flo' or '*.h5'
        flo_flag = -1
        if(suffix == '.flo'):
            flo_flag = 1
        elif(suffix == '.h5'):
            flo_flag = 0
        else:
            assert(False)
            
        subjects = os.listdir(args.gt_path + v)
        for sub in subjects:
     
            txtloc = args.gt_path + v + '/'+ sub
            outloc =  outputfolder + sub
            
            
            
            arffdata = arff.load(open(txtloc, 'r'))      
            first_line = arffdata['data'][0]
            str_label_idx = []
            for i in range(0, len(first_line)):
                if isinstance(first_line[i], str):
                    str_label_idx.append(i)
                
            if(len(str_label_idx) == 0):
                data = np.array(arffdata['data'], dtype = 'float32')
            else:
                data = np.array(arffdata['data'])
                for i in str_label_idx:
                    data[:, i] = list(map(lambda x: translate_dict[x], data[:,i]))
                data = data.astype('float32')
            
            tmp_str = ''  
            with open(txtloc, "r") as f:
                for line in f:
                    if line.startswith("%@METADATA"):
                        tmp_str += line[1:]
                arffdata['description'] = tmp_str
            
            attributes = arffdata['attributes']
            
            timeline = data[:, find_listoftuple(attributes, 'time')]
            
            timeline = timeline - timeline[0]
            
            alldata = np.copy(data)
            
            #preprocessing x and y
            if(args.pre_processing):
                preprocessed_x, preprocessed_y = preproc.preprocessing(data[:, find_listoftuple(attributes, 'x')], data[:, find_listoftuple(attributes, 'y')], 250, 0.0374)
                alldata[:,find_listoftuple(attributes, 'x')] = preprocessed_x
                alldata[:,find_listoftuple(attributes, 'y')] = preprocessed_y
            alldatalist = []
            

            if(flo_flag):
                length = len(os.listdir(of_path))
            else:
                length = len(vidlist)
            
            for f in np.arange(0, length-1):
                partdata = alldata[np.where((timeline>=1e6*f/fps)&(timeline<1e6*(f+1)/fps))]
                partdata2 = alldata[np.where((timeline>=1e6*(f+1)/fps)&(timeline<1e6*(f+2)/fps))]
                #calculate the mean position
                tmp_posi = np.mean(partdata[:, [find_listoftuple(attributes, 'x'), find_listoftuple(attributes, 'y')]], axis=0)
                tmp_posi2 = np.mean(partdata2[:, [find_listoftuple(attributes, 'x'), find_listoftuple(attributes, 'y')]], axis = 0)
    
    
                with h5py.File(mask_path, "r") as k:
                    mask = k['data'][f]
                vel = tmp_posi2 - tmp_posi

                if(flo_flag):                   
                    path_of = of_path + "{:0>5d}".format(f) + ".flo"
                    flow = read(path_of)
                else:
                    path_of = of_path + 'flow.h5'
                    with h5py.File(path_of, "r") as k:
                        flow = k['data'][f]
                
                
                # matching the target, either in velocity space or in pixel space
                if(args.velocity_space):
                    flow = flow_calculation.find_flow_similar(mask,  tmp_posi, partdata.shape[0], vel, flow)
                else:
                    flow = flow_calculation.find_flow(mask,  tmp_posi, partdata.shape[0], flow)
                    
                coord_flow = np.hstack((partdata, np.array(flow).T))

                alldatalist.extend(coord_flow)
 
            alldata_with_flow = np.copy(alldatalist)  
            #feature scale for direction difference computation
            finaldata = slidingwindow(alldata_with_flow[:,find_listoftuple(attributes, 'x')], alldata_with_flow[:,find_listoftuple(attributes, 'y')], alldata_with_flow[:,-2], alldata_with_flow[:,-1], args.feature_scales)
           
            res = np.hstack((alldata_with_flow[4:alldata_with_flow.shape[0]-4][:,:-2],finaldata[:,2:]))
            
            # add attributes and save the arff files    
            if(len(str_label_idx) != 0):
                for idx in str_label_idx:
                    tuple_label = (attributes[idx][0], 'INTEGER')
                    attributes[idx] = tuple_label
            
            flow_ref = [('flow_x', 'NUMERIC'), ('flow_y', 'NUMERIC'),]
            
            for i in args.feature_scales:
                flow_ref.append(('dir_dis_' + str(i), 'NUMERIC' ))
            
            
            attributes += flow_ref
           
            arffdata['data'] = res
            arffdata['attributes'] = attributes
            f = open(outloc, 'w')
            arff.dump(arffdata, f)
            f.close()
            print( outloc +' is finished')
           
        


        
def __main__():
    args = parse_arguments()
    run(args)




if __name__ == '__main__':
    __main__()        
        
        
        
        