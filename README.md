# Improving gaze event classification using computer vision

This is the implementation of the master thesis project for eye movement classfication improved with computer vision techniques.


# DESCRIPTION

This work is based on the previous work of [this paper](https://link.springer.com/article/10.3758/s13428-018-1144-2), we applied the same architecutre but with additional motion of the target as input features.

Before being processed by the model, the data is pre-processed to extract speed, direction, and acceleration features as well as the direction difference, object speed. 

Here is the `default` architecture of their work(as introduced in the paper): 

![alt text](https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/network.png "Model architecture")

Based on their architecture, we added an additional BLSTM layer.


# DEPENDENCIES

Same as in their paper, to make use of this software, you need to first install the [sp_tool](https://github.com/MikhailStartsev/sp_tool/). For its installation instructions see respective README!

If you want to use blstm_model.py script (to train/test models on GazeCom -- data to be found [here](http://michaeldorr.de/smoothpursuit/)), provide the correct path to the sp_tool folder via the `--sp-tool-folder /path/to/sp_tool/` argument.





## Standard package dependencies

* liac-arff 2.4.0: reading and writing the raw gaze data stored in arff files.
* keras 2.4.3 with tensorflow 2.2.0 backend : network building, training as well aspredicting.
* Numpy 1.19.1: regular computing.
* h5py 2.10.0: reading and saving models

# USAGE

1. First, target motion extraction needs the previous knowledge of the instance segmentation masks as well as the optical flow, possible ways are  [Mask-RCNN](https://github.com/facebookresearch/Detectron) and [PWC-Net](https://github.com/NVlabs/PWC-Net), the masks need to be saved as `*.h5` files and the optical flow can be saved either as `*.flo` or as `*.h5` files. In our case, we set the path to save the masks as `/video_name/final_mask_arr.h5`, the flow can be save either as `/video_name/flow.h5` to save all the optical flow across the video or `/video_name/00001.flo` for the optical flow within each frame. But these also can be changed in line 199 and 200.

2. By specifying the arguments of mask path, video path, optical flow path, ground truth path as well as the outputpath, the output `*.arff` files will be created within the output folder. Afterwards, these files can be further used for the matlab files same as in the baseline method. 


