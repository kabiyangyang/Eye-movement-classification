% AnnotateData.m
%
% This function gets ARFF data as input and annotates them with with speed, acceleration, and direction
% attributes at window sizes of 1, 2, 4, 8 and 16 samples width
%
% input:
%   arffFile    - file containing arff data
%   outputFile  - fiel to store data
function AnnotateData(arffFile, outputFile)
    addpath('arff_utils')

    %windowsSize = [1 2 4 8 16];
    windowsSize = [2 4 8 16 32];
    [data, metadata, attributes, relation, comments] = LoadArff(arffFile);
    comments{end+1} = 'The number after speed, direction denotes the step size that was used for the calculation.';
    comments{end+1} = 'Acceleration was calculated between two adjacent samples of the already low pass filtered velocity';

    for i=1:length(windowsSize)
        step = windowsSize(i);

        [speed, direction flowspeed flowdirection vel_distance] = GetVelocity(data, attributes, step);

        speedAttName = ['speed_' num2str(step)];
        [data, attributes] = AddAttArff(data, attributes, speed, speedAttName, 'numeric');

        dirAttName = ['direction_' num2str(step)];
        [data, attributes] = AddAttArff(data, attributes, direction, dirAttName, 'numeric');
        
        flowspeedAttName = ['flow_speed_' num2str(step)];
        [data, attributes] = AddAttArff(data, attributes, flowspeed, flowspeedAttName, 'numeric');   
        
        flowdirAttName = ['flow_dir_' num2str(step)];
        [data, attributes] = AddAttArff(data, attributes, flowdirection, flowdirAttName, 'numeric'); 
        
        veldisAttName = ['speed_dis_' num2str(step)];
        [data, attributes] = AddAttArff(data, attributes, vel_distance, veldisAttName, 'numeric'); 
              

        acceleration = GetAcceleration(data, attributes, speedAttName, dirAttName, 1);
        accAttName = ['acceleration_' num2str(step)];
        [data, attributes] = AddAttArff(data, attributes, acceleration, accAttName, 'numeric');        
    end

    SaveArff(outputFile, data, metadata, attributes, relation, comments);
end
