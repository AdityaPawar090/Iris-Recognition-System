clc;
clear; 
close all;
disp("STEP 3: TESTING NEW IRIS IMAGE")

load irisModel_final.mat

[file,path] = uigetfile({'*.bmp;*.jpg;*.png'},'Select Iris Image');

if isequal(file,0)
    return
end

img = imread(fullfile(path,file));

if size(img,3)==3
    img = rgb2gray(img);
end

img = imresize(img,[128 128]);

img = medfilt2(img,[3 3]);
img = imgaussfilt(img,1);
img = adapthisteq(img);

hog = extractHOGFeatures(img,'CellSize',[8 8]);

lbp = extractLBPFeatures(img,...
    'Radius',2,...
    'NumNeighbors',16,...
    'CellSize',[32 32]);

feature = [hog lbp];

% normalization
feature = (feature - mu_fusion_norm) ./ sigma_fusion_norm;

% PCA
feature = feature - mu_fusion_pca;
feature = feature * coeff_fusion;
feature = feature(:,1:fusion_dim);

svmResult = predict(svmModel,feature);
knnResult = predict(knnModel,feature);
rfResult = predict(rfModel,feature);

disp("Prediction Results")
disp("SVM: "+string(svmResult))
disp("KNN: "+string(knnResult))
disp("RandomForest: "+string(rfResult))
