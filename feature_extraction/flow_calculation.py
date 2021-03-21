# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:22:35 2020

@author: ThinkPad



"""


import cv2
import numpy as np



def find_flow(mask, tmp_posi, length, flow):
            final_flow_x = []
            final_flow_y = []
            
            fore_pos = np.argwhere(mask!=0)
            dist = np.linalg.norm(np.abs(fore_pos[:, [1,0]] - tmp_posi), axis = 1)
            vel_flow = np.ones(length) 
            
            #dist = dist[dist<200]
                
            if(len(dist)!=0):   
                val = mask[fore_pos[np.argmin(dist)][0], fore_pos[np.argmin(dist)][1]]
                    
                
                flow_x = flow[...,0]
                flow_y = flow[...,1]
                res = flow_x[np.argwhere(mask==val)[:,0], np.argwhere(mask==val)[:,1]]
                res2 = flow_y[np.argwhere(mask==val)[:,0], np.argwhere(mask==val)[:,1]]
                tmp_vel_x = np.mean(res)
                tmp_vel_y = np.mean(res2)
                final_flow_x.extend(vel_flow* tmp_vel_x/length)
                final_flow_y.extend(vel_flow*tmp_vel_y/length)
                
                
            else:
                final_flow_x.extend(vel_flow*0)
                final_flow_y.extend(vel_flow*0)
            
            return final_flow_x, final_flow_y
        
        
        
        
        

def find_flow_similar(mask, tmp_posi, length, vel, flow):
            final_flow_x = []
            final_flow_y = []

            fore_pos = np.argwhere(mask!=0)
            dist = np.linalg.norm(np.abs(fore_pos[:, [1,0]] - tmp_posi), axis = 1)
            vel_flow = np.ones(length) 
            tmp = dist<150
            dist = dist[tmp]
                
            if(len(dist)!=0):   
                tmp_vals = mask[fore_pos[tmp][:,0], fore_pos[tmp][:,1]]
                vals = np.unique(tmp_vals)
                
                          
                flow_x = flow[...,0]
                flow_y = flow[...,1]
                flow_vel = []
                tmp_vel_x = 0
                tmp_vel_y = 0
                if(vals.size != 0):
                    for val in vals:
                        res = flow_x[np.argwhere(mask==val)[:,0], np.argwhere(mask==val)[:,1]]
                        res2 = flow_y[np.argwhere(mask==val)[:,0], np.argwhere(mask==val)[:,1]]
                        tmp_vel_x = np.mean(res)
                        tmp_vel_y = np.mean(res2)
                        flow_vel.append([tmp_vel_x, tmp_vel_y])
                
                    vel_dist = np.linalg.norm(flow_vel - vel, axis = 1)
                    res_flow_x = flow_vel[np.argmin(vel_dist)][0]
                    res_flow_y = flow_vel[np.argmin(vel_dist)][1]
                    tmp_val = vals[np.argmin(vel_dist)]
                    final_flow_x.extend(vel_flow* res_flow_x /length)
                    final_flow_y.extend(vel_flow* res_flow_y/length)
                    
                else:
                    final_flow_x.extend(vel_flow*0)
                    final_flow_y.extend(vel_flow*0)
 
                    
                
            else:
                final_flow_x.extend(vel_flow*0)
                final_flow_y.extend(vel_flow*0)

            
            return final_flow_x, final_flow_y
    
    
    
    
    
import numpy as np
import os

TAG_FLOAT = 202021.25

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