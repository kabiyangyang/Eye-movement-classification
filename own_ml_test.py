# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:56:30 2020

@author: ThinkPad
"""


import numpy as np
#import os
import json
import glob
from sp_tool.arff_helper import ArffHelper
import math
from classification import Classification
import warnings
from sklearn import preprocessing

features_to_use = ['speed','flow_speed', 'dir_dis']
feature_files = 'data/inputs/GazeCom_all_features'
gc = json.load(open('data/inputs/GazeCom_video_parameters.json'))
all_video_names = gc['video_names']


def get_arff_attributes_to_keep(features):
    keys_to_keep = []
    if 'xy' in features:
        keys_to_keep += ['x', 'y']
    if 'speed' in features:
        keys_to_keep += ['speed_{}'.format(i) for i in (1, 2, 4, 8, 16)[4:5]]
    if 'direction' in features:
        keys_to_keep += ['direction_{}'.format(i) for i in (1, 2, 4, 8, 16)[:5]]
    if 'acc' in features:
        keys_to_keep += ['acceleration_{}'.format(i) for i in (1, 2, 4, 8, 16)[:5]]
    
    if 'dir_dis' in features:
        keys_to_keep += ['dir_dis_{}'.format(i) for i in (8, 16, 24, 32)[:4]]
    
    if 'flow_speed' in features:
        keys_to_keep += ['flow_speed_{}'.format(i) for i in (1, 2, 4, 8, 16)[:4]]
        
    if 'speed_dis' in features:
        keys_to_keep += ['speed_dis_{}'.format(i) for i in (1, 2, 4, 8, 16)[:5]]

    return keys_to_keep




def calculate_ppd(arff_object, skip_consistency_check=False):
    """
    Pixel-per-degree value is computed as an average of pixel-per-degree values for each dimension (X and Y).

    :param arff_object: arff object, i.e. a dictionary that includes the 'metadata' key.
                @METADATA in arff object must include "width_px", "height_px", "distance_mm", "width_mm" and
                "height_mm" keys for successful ppd computation.
    :param skip_consistency_check: if True, will not check that the PPD value for the X axis resembles that of
                                   the Y axis
    :return: pixel per degree.

    """
    # Previous version of @METADATA keys, now obsolete
    calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING = {
        'PIXELX': ('width_px', lambda val: val),
        'PIXELY': ('height_px', lambda val: val),
        'DIMENSIONX': ('width_mm', lambda val: val * 1e3),
        'DIMENSIONY': ('height_mm', lambda val: val * 1e3),
        'DISTANCE': ('distance_mm', lambda val: val * 1e3)
    }

    for obsolete_key, (new_key, value_modifier) in calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING.items():
        if obsolete_key in arff_object['metadata'] and new_key not in arff_object['metadata']:
            warnings.warn('Keys {} are obsolete and will not necessarily be supported in future. '
                          'Consider using their more explicit alternatives: {}'
                          .format(calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING.keys(),
                                  [val[0] for val in calculate_ppd.OBSOLETE_METADATA_KEYS_MAPPING.values()]))
            # replace the key
            arff_object['metadata'][new_key] = value_modifier(arff_object['metadata'].pop(obsolete_key))

    theta_w = 2 * math.atan(arff_object['metadata']['width_mm'] /
                            (2 * arff_object['metadata']['distance_mm'])) * 180. / math.pi
    theta_h = 2 * math.atan(arff_object['metadata']['height_mm'] /
                            (2 * arff_object['metadata']['distance_mm'])) * 180. / math.pi

    ppdx = arff_object['metadata']['width_px'] / theta_w
    ppdy = arff_object['metadata']['height_px'] / theta_h

    ppd_relative_diff_thd = 0.2
    if not skip_consistency_check and abs(ppdx - ppdy) > ppd_relative_diff_thd * (ppdx + ppdy) / 2:
        warnings.warn('Pixel-per-degree values for x-axis and y-axis differ '
                      'by more than {}% in source file {}! '
                      'PPD-x = {}, PPD-y = {}.'.format(ppd_relative_diff_thd * 100,
                                                       arff_object['metadata'].get('filename', ''),
                                                       ppdx, ppdy))
    return (ppdx + ppdy) / 2




total_files = 0
files_template = feature_files + '/{}/*.arff'

data_X = []
data_Y = []
indexs = []
keys_to_keep = get_arff_attributes_to_keep(features_to_use)
CLEAN_TIME_LIMIT = 21 * 1e6
keys_to_convert_to_degrees = ['x', 'y'] + [k for k in keys_to_keep if 'speed_' in k or 'acceleration_' in k or 'flow_speed' in k or 'speed_dis' in k]
keys_to_convert_to_degrees = sorted(set(keys_to_convert_to_degrees).intersection(keys_to_keep))
for video_name in all_video_names:
    print('For {} using files from {}'.format(video_name, files_template.format(video_name)))
    fnames = sorted(glob.glob(files_template.format(video_name)))
    total_files += len(fnames)

    for f in fnames:
        o = ArffHelper.load(open(f, encoding = 'utf-8'))
        if 'SSK_' in f:
            o['data'] = o['data'][::2]
        o['data'] = o['data'][o['data']['time'] <= CLEAN_TIME_LIMIT]
        
        ppd_f = calculate_ppd(o)
        for k in keys_to_convert_to_degrees:
                    o['data'][k] /= ppd_f
                    #o['data'][k] = preprocessing.normalize(o['data'][k])
                    scaler = preprocessing.StandardScaler().fit(o['data'][k].reshape(-1,1))
                    tmp_data_k = scaler.transform(o['data'][k].reshape(-1,1)) 
                    o['data'][k] = tmp_data_k.ravel()
                    
        data_X.extend(np.hstack([np.reshape(o['data'][key], (-1, 1)) for key in keys_to_keep]).astype(np.float64))                    
        data_Y.extend(o['data']['handlabeller_final'])
    indexs.append(len(data_X))
        
dataset = np.hstack((np.copy(data_X), np.copy(data_Y).reshape(-1,1)))
#LOVO classification
indexs.insert(0,0)
for i in np.arange(0, len(indexs)-1):
    
    tmpdata = np.copy(dataset)
    testdata = tmpdata[indexs[i]:indexs[i+1]]
    trainingdata = np.delete(tmpdata, np.arange(indexs[i],indexs[i+1]), axis=0)
    
    clf = Classification('Randomforest', trainingdata)
    clf.classify(testdata)


