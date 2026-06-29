clc; clear; close all;
disp("STEP 2: TRAINING AND COMPARISONS")

load irisFeatures.mat

labels = categorical(labels);

% -------------------
% NORMALIZATION (VERY IMPORTANT)
% -------------------
[features_HOG,mu_hog,sigma_hog] = zscore(features_HOG);
[features_LBP,mu_lbp,sigma_lbp] = zscore(features_LBP);
[features_FUSION,mu_fusion_norm,sigma_fusion_norm] = zscore(features_FUSION);

% -------------------
% PCA
% -------------------
[coeff_hog,score_hog,~,~,~,mu_hog_pca] = pca(features_HOG);
[coeff_lbp,score_lbp,~,~,~,mu_lbp_pca] = pca(features_LBP);
[coeff_fusion,score_fusion,~,~,~,mu_fusion_pca] = pca(features_FUSION);

hog_dim = min(80,size(score_hog,2));
lbp_dim = min(80,size(score_lbp,2));
fusion_dim = min(150,size(score_fusion,2));

features_HOG = score_hog(:,1:hog_dim);
features_LBP = score_lbp(:,1:lbp_dim);
features_FUSION = score_fusion(:,1:fusion_dim);

% -------------------
% TRAIN TEST SPLIT
% -------------------
cv = cvpartition(labels,'HoldOut',0.3);

trainIdx = training(cv);
testIdx = test(cv);

XTrain_HOG = features_HOG(trainIdx,:);
XTest_HOG = features_HOG(testIdx,:);

XTrain_LBP = features_LBP(trainIdx,:);
XTest_LBP = features_LBP(testIdx,:);

XTrain_FUSION = features_FUSION(trainIdx,:);
XTest_FUSION = features_FUSION(testIdx,:);

YTrain = labels(trainIdx);
YTest = labels(testIdx);

% -------------------
% SVM (RBF KERNEL)
% -------------------
disp("Training SVM...")

svmModel = fitcecoc(XTrain_FUSION,YTrain,...
    'Learners',templateSVM('KernelFunction','rbf'));

svmPred = predict(svmModel,XTest_FUSION);
svmAcc = mean(svmPred==YTest)*100;

% -------------------
% KNN
% -------------------
disp("Training KNN...")

knnModel = fitcknn(XTrain_FUSION,YTrain,...
    'NumNeighbors',3,...
    'Distance','cosine');

knnPred = predict(knnModel,XTest_FUSION);
knnAcc = mean(knnPred==YTest)*100;

% -------------------
% RANDOM FOREST
% -------------------
disp("Training Random Forest...")

rfModel = fitcensemble(XTrain_FUSION,YTrain,'Method','Bag');

rfPred = predict(rfModel,XTest_FUSION);
rfAcc = mean(rfPred==YTest)*100;

% -------------------
% FEATURE COMPARISON
% -------------------
model_hog = fitcecoc(XTrain_HOG,YTrain);
pred_hog = predict(model_hog,XTest_HOG);
hogAcc = mean(pred_hog==YTest)*100;

model_lbp = fitcecoc(XTrain_LBP,YTrain);
pred_lbp = predict(model_lbp,XTest_LBP);
lbpAcc = mean(pred_lbp==YTest)*100;

fusionAcc = svmAcc;

FeatureMethods = ["HOG";"LBP";"HOG+LBP"];
FeatureAccuracy = [hogAcc; lbpAcc; fusionAcc];

T1 = table(FeatureMethods,FeatureAccuracy)

Classifier = ["SVM";"KNN";"RandomForest"];
ClassifierAccuracy = [svmAcc; knnAcc; rfAcc];

T2 = table(Classifier,ClassifierAccuracy)

figure
bar(FeatureAccuracy)
set(gca,'XTickLabel',FeatureMethods)
title("Feature Extraction Comparison")
ylabel("Accuracy %")

figure
bar(ClassifierAccuracy)
set(gca,'XTickLabel',Classifier)
title("Classifier Comparison")
ylabel("Accuracy %")

save("irisModel_final.mat","svmModel","knnModel","rfModel","coeff_fusion","fusion_dim","mu_fusion_norm","sigma_fusion_norm","mu_fusion_pca")

disp("Training completed")
