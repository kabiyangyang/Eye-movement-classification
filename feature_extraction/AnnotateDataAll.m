function AnnotateDataAll(arffBasepath, outBasepath)
    %arffBasepath = 'C:/Users/ThinkPad/Downloads/deep_em_classifier-master/deep_em_classifier-master/GazeCom/gaze_arff/'
    arffBasepath = 'D:/deep_em_classifier-master/deep_em_classifier-master/GazeCom/ground_truth_with_flow'
    outBasepath = 'D:/deep_em_classifier-master/deep_em_classifier-master/data/inputs/GazeCom_all_features'
    outputDir = [outBasepath '/'];

    if (exist(outputDir) ~= 7)
        mkdir(outputDir);
    end
    tmp_folders = dir([arffBasepath '/*']);
    folders = {tmp_folders.name};
    folders = folders(3:12);
    
    for i = 1:10
        tmpbasepath = [arffBasepath '/' folders{i}];
        arffFiles = glob([tmpbasepath '/*.arff']);
        
        outputfolder = [outputDir '/' folders{i}];
        if (exist(outputfolder) ~= 7)
            mkdir(outputfolder);
        end
        for arffInd=1:size(arffFiles)
            arffFile = arffFiles{arffInd,1};
            [arffDir, arffName, ext] = fileparts(arffFile);
            outputFile = [outputfolder '/' arffName '.arff'];

            disp(['Processing ' arffFile]);
            AnnotateData(arffFile, outputFile);
        end
    end