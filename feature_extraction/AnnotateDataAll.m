function AnnotateDataAll(arffBasepath, outBasepath)

    arffBasepath = 'D:/processed_data/mn_ra/'
    outBasepath = 'D:/processed_data/mn_ra_processed/'
    
    outputDir = [outBasepath '/'];

    if (exist(outputDir) ~= 7)
        mkdir(outputDir);
    end
    tmp_folders = dir([arffBasepath '/*']);
    folders = {tmp_folders.name};
    folders = folders(3:length(folders));
    
    for i = 1:(length(folders))

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
