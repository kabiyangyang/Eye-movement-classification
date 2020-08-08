% GetVelocity.m
%
% This funciton calcualtes the speed and direction for the given data and step size
%
% input:
%   data        - arff data
%   attributes  - attrbitues describing the data
%   windowWidth - step size
%
% output:
%   speed       - vector containing speed in pixels per second
%   direction   - direction of movement in rands from -pi to pi

function [speed, direction flowspeed flowdirection vel_distance] = GetVelocity(data, attributes, windowWidth)
    c_minConf = 0.75;
    step = ceil(windowWidth/2);

    speed = zeros(size(data,1),1);
    direction = zeros(size(data,1),1);
    flowspeed = zeros(size(data,1),1);
    flowdirection = zeros(size(data,1),1);
    vel_distance = zeros(size(data,1),1);
    
    timeInd = GetAttPositionArff(attributes, 'time');
    xInd = GetAttPositionArff(attributes, 'x');
    yInd = GetAttPositionArff(attributes, 'y');
    flowxInd = GetAttPositionArff(attributes, 'flow_x');
    flowyInd = GetAttPositionArff(attributes, 'flow_y');
    confInd = GetAttPositionArff(attributes, 'confidence');

    for i=1:size(data,1)
        if (data(i,confInd) < c_minConf)
            continue;
        end

        % get initial interval
        if (step == windowWidth)
            startPos = i - step;
            endPos = i;
        else
            startPos = i -step;
            endPos = i + step;
        end

        % fine tune intervals
        if (startPos < 1 || data(startPos,confInd) < c_minConf)
            startPos = i;
        end
        if (endPos > size(data,1) || data(endPos,confInd) < c_minConf)
            endPos = i;
        end

        % invalid interval
        if (startPos == endPos)
            continue;
        end

        ampl = sqrt((data(endPos,xInd)-data(startPos,xInd))^2 + (data(endPos,yInd)-data(startPos,yInd))^2);
        time = (data(endPos,timeInd) - data(startPos,timeInd))/1000000;
        speed(i) = ampl/time;
        
        amplflow = sqrt((sum(data(startPos:endPos-1, flowxInd)))^2 + (sum(data(startPos:endPos-1, flowyInd)))^2);
        flowspeed(i) = amplflow/time;
        
        direction(i) = atan2(data(endPos,yInd)-data(startPos,yInd), data(endPos,xInd)-data(startPos,xInd));
        flowdirection(i) = atan2(sum(data(startPos:endPos-1, flowyInd)), sum(data(startPos:endPos-1, flowxInd)));
        
        vel_distance(i) = flowspeed(i) - speed(i);
   
end
