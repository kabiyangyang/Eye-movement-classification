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

The files in this repository provide the interface to our 1D CNN-BLSTM eye movement classification model.

`blstm_model.py` is the main script that contains all the necessary tools to train and test the model (currently -- on GazeCom).

## Testing the model on your data

If you simply want to run our best (so far) model(s) on external data, this is the pipeline you need to follow (using the `blstm_model_run.py` script, which is a reduced-interface version of the `blstm_model.py`, see below). This is, perhaps, not the final version of the code, but the pipeline (steps 1 to 3) outlined below has been tested.


1. Convert your data into .arff file format (see `example_data/input.arff` for an example file). Your .arff files need to contain some metadata about the eye tracking experiment (the values are to be set in accordance to your particular data set!):

        %@METADATA width_px 1280.0
        %@METADATA height_px 720.0
        %@METADATA width_mm 400.0
        %@METADATA height_mm 225.0
        %@METADATA distance_mm 450.0

    The fields indicate 
    * the width and height of the stimulus or monitor (in pixels), 
    * the width and height of the stimulus or monitor in millimeters,
    * the distance between the subjects' eyes and the stimulus or monitor (in millimeters).

    The data in this file needs just 4 columns:
    * time (in microseconds)
    * x and y - the on-screen (or relative to the stimulus) coordinates (in pixels; will be converted to degrees of visual angle automatically - that is why we need the metadata)
    * confidence - the eye tracker confidence for the tracking of the subjects' eyes (1 means normal tracking, 0 means lost tracking). If you do not have this information, set to 1 everywhere.

    The format is better described in https://ieeexplore.ieee.org/abstract/document/7851169/ 

2. Such files can then be processed by the Matlab scripts in the `feature_extraction` folder to produce .arff files with features of the gaze (see an example file in example_data/features.arff). For single files, use AnnotateData.m (see respective README; usage: `AnnotateData('path/to/input/file.arff', 'path/to/output/file.arff')` ). You can alternatively use the model that only utilises x and y as feature, in which case you can skip this step.

3. Call `python blstm_model_run.py --input path/to/extracted/feature/file.arff --output folder/where/to/put/the/result.arff --model path/to/model/folder/or/file --feat <feature group name> <feature group name>`.

      Feature groups refer to the groups of features that are present in the path/to/extracted/feature/file.arff file. Can be the following: xy, speed, direction, acceleration. Any number of feature groups can be listed (use space as a separator, see example below). We found that using speed and direction as features performed best.

      Example command: 
> python blstm_model_run.py --feat speed direction --model example_data/model.h5 --in example_data/features.arff --out example_data/output_reproduced.arff

More info about the arguments can be obtained via 

> python blstm_model_run.py --help

## Training and testing a new model

For this purpose, use the `blstm_model.py` script. You can find out all its options by running

> python blstm_model.py --help

If you wish to train on another data set or to use some new features, please take note of the comments in `blstm_model.py`, in particular in the beginning of the `run()` function.

During training (after each cross-validation fold has been processed), the script will output some sample-level performance statistics (F1 scores) for the detection of fixation, saccade, pursuit, and noise samples. *These results are intermediate and serve the monitoring purposes only!* The full evaluation can (and will) only be undertaken when all the cross-validation folds are processed. 

### Multi-instance training (recommended)

For training, we usually used the `--run-once` option, which will only train one model (i.e. run thorough only one fold of the Leave-One-Video-Out cross-validation process), since some GPU memory is likely not perfectly freed, and this allows for simultaneous training on several program instance or machines (for different folds of our cross-validation), provided that they have access to some common folder for synchronisation. 

Each run will first create a placeholder .h5 model file and then run the training. The model training for the folds with already existing corresponding .h5 files is omitted. With this option the script needs to be run 18 times (same as the number of videos in GazeCom), so a simplest bash-wrapper would be handy. No additional parameters, which specify the fold of the cross-validation is to be processed, are necessary (although it can be specified through `--run-once-video`). 

If you are running the training on several machines or script instances, you can run the 18-iteration loop of the `--run-once`-enabled commands, and the next free machine will start processing the next needed cross-validation fold.

One thing to note here is that is you interrupt the training process, you will end up with one or more empty .h5 files in the corresponding model folder. **These need to be deleted before the training is resumed**, since the training on these folds will be skipped otherwise.

#### NB. Right now the path to storing models (`data/models`) and outputs (`data/outputs`) are set for convenient local training. Set model paths to a location that is accessible to all instances of the program that you plan to run by setting an appropriate `--model-root-path`. This folder might need to be created beforehand.

### Other important options

* `--final-run` is for inference run only. This disables the data pre-cleaning when loading it. This operation mode is intended for getting the final classification results with a set of already trained models
* `--output-folder` is the output folder, where the labelled .arff files will be saved. Set to `auto` if you wish for the folder to be selected automatically (the folder name will include the model structure descriptor). **If no argument is provided, no outputs will be created!**
* `--model-root-path` is the path, where all the models will be stored. The script will create a sub-folder for each model architecture (the name will contain the model descriptor), and this sub-folder will then contain individual .h5 trained model files for each of the cross-validation folds when the respective iteration finishes.
* `--batch-size`, `--num-epochs`, `--initial-epoch` (useful for fine-tuning a model), `--training-samples` (**mind the help if adjusting this option**: you might need to adjust `--overlap`, too) are all fairly standard parameters
* since the model deals with temporal windows of data (we tested windows up to 257 samples), the parameters for such windows can be specified via `--window-size` (number of samples) and `--window-overlap` (overlap between the sampled windows -- don't necessarily want to have windows that are only shifted by 1 sample, could lead to overfitting). Generally, using larger window sizes leads to better performance (larger context = better classification):

<p align="center"><img src="https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/performance_vs_context.png" width="400"></p>

* `--features` is another important parameter. It lists the gaze coordinates' features that you want to use for training. It supports the following options: `xy`, `speed`, `direction`, `acceleration`, and `movement`, the latter referring to the speed, direction, and acceleration features combined. Features can be specified in combination, e.g. `--features speed acceleration`. In our tests, using acceleration decreases the overall performance of the model, especially for smooth pursuit.


### Architecture-defining options

You can configure the architecture by passing appropriate parameters via console arguments. Any architecture that can be achieved by the use of these options will consist of 3 blocks (which can also be omitted):
* Convolutional
  * `--num-conv` will set the number of convolutional layers (default: 3, **recommended: 4**)
  
  <p align="center"><img src="https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/num_conv.png" width="400"></p>
  
  * `--conv-padding-mode` sets the padding mode (valid or same)
  * `--conv-units` can be used to set the number of convolutional filters that are learned on each layer. This parameter accepts a list of values (e.g. `--conv-units 32 16 8`). If the list is longer than `--num-conv`, it will be truncated. If it is shorter -- the last element is repeated as many times as necessary, so passing `--conv-units 32 16 8` together with `--num-conv 5` will result in 5 convolutional layers, with 32, 16, 8, 8, and 8 filters, respectively

* Dense (fully-connected)
  * `--num-dense` sets the number of dense layers (default: 1, **recommended: 0**)  
  * `--dense-units` acts much like `--conv-units`, but for the number of dense units in respective layers
  
  Not using any dense layers proved to be a better choice:
  
  <p align="center"><img src="https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/num_dense.png" height="300"> <img src="https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/dense_units.png" height="300"></p>
  
* BLSTM
  * `--num-blstm` sets the number of BLSTM layers (default: 1, **recommended: 2**)
  
  <p align="center"><img src="https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/num_blstm.png" width="400"></p>
  
  * `--blstm-units` acts just like `--conv-units`, but for the number of BLSTM units in respective layers
  * `--no-bidirectional` will force the model to use LSTM instead of BLSTM (leasd to poorer performance, but could be used in an online detection set-up). The plot below represents the training loss value (categorical cross-entropy) for BLSTM vs 2 stacked uni-directional LSTMs (to roughly match the number of parameters) models:
  
  <p align="center"><img src="https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/blstm_vs_lstm.png" width="400"></p>

Here is the comparison between the achieved F1 scores of our "default" architecture and the "recommended" final one:

![alt text](https://github.com/MikhailStartsev/deep_em_classifier/blob/master/figures/performance_ours.png "Comparison to of the default and the revised architectures")
  
If you want to just create the architecture and see the number of trainable parameters or other details, use `--dry-run`.
