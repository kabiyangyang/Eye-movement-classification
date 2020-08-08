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

def parse_arguments():

    parser = ArgumentParser('flow feature processing')

    parser.add_argument('--gt-path',  default='../GazeCom/ground_truth/',
                        help='The path that contains the ground truth data with hand labeled classes')
    parser.add_argument('--flow-path', default='F:/of2/',
                        help='The path that contains the flow data')

    parser.add_argument('--mask-path', default='E:/movies-m2t/',
                        help='Folder containing the files with already extracted features.')
    parser.add_argument('--output-path',  default='../GazeCom/ground_truth_with_flow/',
                        help='Folder containing the ground truth.')
    
    parser.add_argument('--feature-scales', nargs='+', default=[8, 16, 24, 32], type=int,
                        help='temproal scales for the flow and gaze moving direction distances')
    
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
    Param window_width_interval: user defined temproal scales for flow 
    and gaze moving direction distances
    """

    #t = arr_b_c[5:len(arr_b_c)-4]
    x = moving_average_2(arr_x, 10)
    y = moving_average_2(arr_y, 10)
    flow_x = moving_average_2(fx, 10)
    flow_y = moving_average_2(fy, 10)
    
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
        
        d_res[d_res>180] = 360 - d_res[d_res>180]
        d_res[d_res<-180] = (-360) - d_res[d_res<-180]
        
        
        outputdata = np.hstack((outputdata, d_res.reshape(-1,1)))

    return outputdata


def run(args):
    """
    run function: for each video and each participants, find out the nearest object and apply
    object mask on the flow data, then preprocessing the data and save them as featrues into the .arff files.
    
    """
    
    fps = 30
    videos_run = ['koenigstrasse','holsten_gate', 'puppies', 'roundabout', 'sea']#= os.listdir(args.flow_path)
    
    for v in videos_run:
        
        videoloc = args.mask_path + v + '.m2t'
        outputfolder = args.output_path + v + '/'
        
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
        subjects = os.listdir(args.gt_path + v) 
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
        mask_path = args.mask_path +'/'+ v + '/seg/final_mask_arr.npy'
        mask_all = np.load(mask_path, allow_pickle = True)
        for sub in subjects:
            s = sub[0:4]
            txtloc = args.gt_path + v + '/'+ s + v+'.arff'
            outloc =  outputfolder + s + v+ '.arff'
            
            data = arff.load(open(txtloc, 'r'))
        
            alldata = np.array(data['data'])
            timeline = alldata[:,0]
            alldatalist = []
            of_path = args.flow_path +'/'+ v +'/'
            length = len(os.listdir(of_path))
            for f in np.arange(length):
                partdata = alldata[np.where((timeline>1e6*f/fps)&(timeline<1e6*(f+1)/fps))]
                tmp_posi = np.mean(partdata[:,1:3], axis=0)
                mask = mask_all[f]
              
                fx, fy = flow_calculation.find_flow(mask, of_path, f, tmp_posi, partdata[:,1:3].shape[0])
                tmp_flow = np.vstack((fx, fy)).T
                
                partdatawithflow = np.hstack((partdata, tmp_flow))
    
                alldatalist.extend(partdatawithflow)
            test = np.copy(alldatalist)
            cl = test[:,6]
                
            preprocessed_x, preprocessed_y = preproc.preprocessing(cl, test[:,1], test[:,2], 250, 0.0374)
            test[:,1] = np.copy(preprocessed_x)
            test[:,2] = np.copy(preprocessed_y)

            
                
            finaldata = slidingwindow(test[:,1], test[:,2], test[:,7], test[:,8], args.feature_scales)
            
            #bc moving average filter length was set 10, here was set 10 elements less accordingly
            to_be_input_data = test[5:test.shape[0]-4][:,:-2]
            
            res = np.hstack((to_be_input_data,finaldata[:,2:]))
                
            
            tmp_ref = data['attributes']
            flow_ref = [('flow_x', 'NUMERIC'), ('flow_y', 'NUMERIC'), ('dir_dis_8', 'NUMERIC'), ('dir_dis_16', 'NUMERIC') ,('dir_dis_24', 'NUMERIC'), ('dir_dis_32', 'NUMERIC')]
            tmp_ref.extend(flow_ref)
           
            data['data'] = res
            data['attributes'] = tmp_ref
            f = open(outloc, 'w')
            arff.dump(data, f)
            f.close()
            print( outloc +' is finished')
           
        
                
        del mask_all
    return preprocessed_x, preprocessed_y
        
        
def __main__():
    args = parse_arguments()
    m1, m2 = run(args)
    return m1, m2
if __name__ == '__main__':
   m1, m2 =  __main__()        
        
        
        
        