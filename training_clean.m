clc; clear; close all;
disp("STEP 1: FEATURE EXTRACTION FOR MMU IRIS DATASET")

datasetPath = "MMU-Iris-Database";
targetSize = [128 128];

features_HOG = [];
features_LBP = [];
features_FUSION = [];
labels = [];

persons = dir(datasetPath);
persons = persons([persons.isdir]);
persons = persons(~ismember({persons.name},{'.','..'}));

for p = 1:length(persons)

    personID = persons(p).name;
    personPath = fullfile(datasetPath,personID);

    eyeFolders = dir(personPath);
    eyeFolders = eyeFolders([eyeFolders.isdir]);
    eyeFolders = eyeFolders(~ismember({eyeFolders.name},{'.','..'}));

    for e = 1:length(eyeFolders)

        eyePath = fullfile(personPath,eyeFolders(e).name);
        images = dir(fullfile(eyePath,'*.bmp'));

        for i = 1:length(images)

            imgPath = fullfile(eyePath,images(i).name);
            img = imread(imgPath);

            if size(img,3)==3
                img = rgb2gray(img);
            end

            img = imresize(img,targetSize);

            % Preprocessing
            img = medfilt2(img,[3 3]);
            img = imgaussfilt(img,1);
            img = adapthisteq(img);

            % HOG
            hog = extractHOGFeatures(img,'CellSize',[8 8]);

            % IMPROVED LBP
            lbp = extractLBPFeatures(img,...
                'Radius',2,...
                'NumNeighbors',16,...
                'CellSize',[32 32]);

            fusion = [hog lbp];

            features_HOG = [features_HOG ; hog];
            features_LBP = [features_LBP ; lbp];
            features_FUSION = [features_FUSION ; fusion];

            labels = [labels ; str2double(personID)];

        end
    end
end

save("irisFeatures.mat","features_HOG","features_LBP","features_FUSION","labels")

disp("Feature extraction completed")
disp("Total images processed: "+num2str(length(labels)))
