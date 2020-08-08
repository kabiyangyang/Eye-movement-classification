# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:21:09 2020

@author: ThinkPad
"""

import pandas as pd
import numpy as np
from statsmodels.robust.scale import mad
from scipy import signal
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from math import (
    degrees,
    atan2,
)
import logging


lgr = logging.getLogger('feature_extraction.preproc')
def preprocess(
            sr,
            px2deg,
            data,
            min_blink_duration=0.02,
            dilate_nan=0.01,
            median_filter_length=0.05,
            savgol_length=0.14,
            savgol_polyord=2,
            max_vel=1000.0):
        """
        Parameters
        ----------
        min_blink_duration : float
          In seconds. Any signal loss shorter than this duration will not be
          considered for `dilate_nan`.
        dilate_nan : float
          Duration by which to dilate a blink window (missing data segment) on
          either side (in seconds).
        median_filter_length : float
          Filter window length in seconds.
        savgol_length : float
          Filter window length in seconds.
        savgol_polyord : int
          Filter polynomial order used to fit the samples.
        max_vel : float
          Maximum velocity in deg/s. Any velocity value larger than this
          threshold will be replaced by the previous velocity value.
          Additionally a warning will be issued to indicate a potentially
          inappropriate filter setup.
        """
        # convert params in seconds to #samples
        dilate_nan = int(dilate_nan * sr)
        min_blink_duration = int(min_blink_duration * sr)
        savgol_length = int(savgol_length * sr)
        median_filter_length = int(median_filter_length * sr)

        # in-place spike filter
        data = filter_spikes(data)

        # for signal loss exceeding the minimum blink duration, add additional
        # dilate_nan at either end
        # find clusters of "no data"
        if dilate_nan:
            lgr.info('Dilate NaN segments by %i samples', dilate_nan)
            mask = get_dilated_nan_mask(
                data['x'],
                dilate_nan,
                min_blink_duration)
            data['x'][mask] = np.nan
            data['y'][mask] = np.nan

        if savgol_length:
            lgr.info(
                'Smooth coordinates with Savitzy-Golay filter (len=%i, ord=%i)',
                savgol_length, savgol_polyord)
            for i in ('x', 'y'):
                data[i] = savgol_filter(data[i], savgol_length, savgol_polyord)

        # velocity calculation, exclude velocities over `max_vel`
        # no entry for first datapoint!
        velocities = _get_velocities(px2deg, sr, data)

        # replace "too fast" velocities with previous velocity
        # add missing first datapoint

        for vel in velocities:
            if vel > max_vel:  # deg/s
                # ignore very fast velocities
                lgr.warning(
                    'Computed velocity exceeds threshold. '
                    'Inappropriate filter setup? [%.1f > %.1f deg/s]',
                    vel,
                    max_vel)
                
        return data


def get_dilated_nan_mask(arr, iterations, max_ignore_size=None):
    clusters, nclusters = ndimage.label(np.isnan(arr))
    # go through all clusters and remove any cluster that is less
    # the max_ignore_size
    for i in range(nclusters):
        # cluster index is base1
        i = i + 1
        if (clusters == i).sum() <= max_ignore_size:
            clusters[clusters == i] = 0
    # mask to cover all samples with dataloss > `max_ignore_size`
    mask = ndimage.binary_dilation(clusters > 0, iterations=iterations)
    return mask


def filter_spikes(data):
    """
    spikes filer function,  for all neighboring 3 values, check the implausible spikes,
    assign the closest value to the 2nd value 
    
    """
    def _filter(arr):
        # over all triples of neighboring samples
        for i in range(1, len(arr) - 1):
            if (arr[i - 1] < arr[i] and arr[i] > arr[i + 1]) \
                    or (arr[i - 1] > arr[i] and arr[i] < arr[i + 1]):
                # immediate sign-reversal of the difference from
                # x-1 -> x -> x+1
                prev_dist = abs(arr[i - 1] - arr[i])
                next_dist = abs(arr[i + 1] - arr[i])
                # replace x by the neighboring value that is closest
                # in value
                arr[i] = arr[i - 1] \
                    if prev_dist < next_dist else arr[i + 1]
        return arr

    data['x'] = _filter(data['x'])
    data['y'] = _filter(data['y'])
    return data


def preprocessing(vel_all, vel_gaze_labeled, vel_gaze_labeled_2, sr, px2deg):

    df = pd.DataFrame()

    x = np.copy(vel_gaze_labeled)
    y = np.copy(vel_gaze_labeled_2)

    df['x'] = np.copy(x)
    df['y'] = np.copy(y)

    new_df = preprocess(sr, px2deg, df)


    return np.copy(new_df['x']), np.copy(new_df['y'])


def _get_velocities(px2deg, sr , data):
        # euclidean distance between successive coordinate samples
        # no entry for first datapoint!
        velocities = (np.diff(data['x']) ** 2 + np.diff(data['y']) ** 2) ** 0.5
        # convert from px/sample to deg/s
        velocities *= px2deg * sr
        return velocities
