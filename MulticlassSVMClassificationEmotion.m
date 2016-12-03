% SVM Mulitclass Classification for emotions
% This performs SVM Multiclass  Classification using one vs rest methodology
% For each binary learner, one class is positive and the rest are negative. 
% This design exhausts all combinations of positive class assignments.
clear; clc;
addpath('FeatureFiles\');


%% Performing Emotion Recognition on the first combination of features 
% (a)Haar+ Gabor+Morphological boundary
% Loading the Combined Features

load('FeatureFiles\CombinationFeaturesHaarGaborMorph.mat');
load('EmotionsLabels.mat');

% Shuffle the training set 

idx = randperm(length(Emotion_label));
train_idx = round(0.90*size(Emotion_label, 1));
% Performing Training on 90% of the dataset and Testing on 10% of the
% dataset
feature_vect_train = featureVectHaarGaborMorph(idx(1:train_idx),:);
class_train = Emotion_label(idx(1:train_idx),:);
SVMMulticlass_model_a = fitcecoc(feature_vect_train, class_train, 'Coding', 'onevsall',...
                                    'Learners', 'svm'); 
trainLabel_emotion = predict(SVMMulticlass_model_a, feature_vect_train);
target_emotion = Emotion_label(idx(train_idx+1:end),:);
feature_vect_test = featureVectHaarGaborMorph(idx(train_idx+1:end),:);

predictLabel_emotion = predict(SVMMulticlass_model_a, feature_vect_test);

% Computing Training Error
train_loss = eval_mcr(trainLabel_emotion, class_train);
% Computing Testing Error
test_loss = eval_mcr(predictLabel_emotion, target_emotion);
%[conf_HGM, confmat_HGM, ind_HGM, error_HGM] = confusion(target_emotion, predictLabel_emotion);

% Performing 10 fold cross vaalidation
feature_comba = featureVectHaarGaborMorph(idx(1:end),:);
emotion_labels = Emotion_label(idx(1:end),:);
SVMMulticlass_a = fitcecoc(feature_comba, emotion_labels, 'Coding', 'onevsall',...
                                    'Learners', 'svm'); 

HaarGaborMorphCrossVal = crossval(SVMMulticlass_a, 'KFold', 10);
cross_val_comba_error = kfoldLoss(HaarGaborMorphCrossVal);
fprintf('The 10 fold cross-validation error using Haar+ Gabor+Morphological boundary features = %f\n', cross_val_comba_error);

%HaarGaborMorphCombo = struct('conf_HGM',conf_HGM,'confmat_HGM', confmat_HGM, 'ind_HGM', ind_HGM, ...
%                             'error_HGM',error_HGM, 'cross_val_HGM', cross_val_comba_error, 'features', 'HaarGaborMorph');
%save('HaarGaborMorphClassifyResults.mat', 'HaarGaborMorphCombo');                         


%% Performing Emotion Recognition on the second combination of features 
% (b)Eigenfaces + Gabor + Haar
% Loading the Combined Features
clear;
load('FeatureFiles\CombinationFeaturesHaarGaborEigen.mat');
load('EmotionsLabels.mat');

% Shuffle the training set 

idx = randperm(length(Emotion_label));
train_idx = round(0.90*size(Emotion_label, 1));
% Performing Training on 90% of the dataset and Testing on 10% of the
% dataset
feature_vect_train = featureVectHaarGaborEigenface(idx(1:train_idx),:);
class_train = Emotion_label(idx(1:train_idx),:);
SVMMulticlass_model_b = fitcecoc(feature_vect_train, class_train, 'Coding', 'onevsall',...
                                    'Learners', 'svm'); 
trainLabel_emotion = predict(SVMMulticlass_model_b, feature_vect_train);

target_emotion = Emotion_label(idx(train_idx+1:end),:);
feature_vect_test = featureVectHaarGaborEigenface(idx(train_idx+1:end),:);
predictLabel_emotion = predict(SVMMulticlass_model_b, feature_vect_test);

% Computing Training Error
train_loss = eval_mcr(trainLabel_emotion, class_train);
% Computing Testing Error
test_loss = eval_mcr(predictLabel_emotion, target_emotion);
%[conf_HGE, confmat_HGE, ind_HGE, error_HGE] = confusion(target_emotion, predictLabel_emotion);

% Performing 10 fold cross vaalidation
feature_combb = featureVectHaarGaborEigenface(idx(1:end),:);
emotion_labels = Emotion_label(idx(1:end),:);
SVMMulticlass_b = fitcecoc(feature_combb, emotion_labels, 'Coding', 'onevsall',...
                                    'Learners', 'svm'); 

HaarGaborEigenCrossVal = crossval(SVMMulticlass_b, 'KFold', 10);
cross_val_combb_error = kfoldLoss(HaarGaborEigenCrossVal);
fprintf('The 10 fold cross-validation error using Haar+ Gabor+Eigenfaces features = %f\n', cross_val_combb_error);

%HaarGaborEigenfacesCombo = struct('conf_HGE',conf_HGE,'confmat_HGE', confmat_HGE, 'ind_HGE', ind_HGE, ...
%                             'error_HGE',error_HGE, 'cross_val_HGE', cross_val_combb_error, 'features', 'HaarGaborEigenfaces');
%save('HaarGaborEigenfacesClassifyResults.mat', 'HaarGaborEigenfacesCombo');                         

%% Performing Emotion Recognition on the third combination of features 
% (c) Haar+ Eigenfaces+ Landmark
% Loading the Combined Features
clear;
load('FeatureFiles\CombinationFeaturesLandmarkHaarEigen.mat');
load('EmotionsLabels.mat');
% Shuffle the training set 

idx = randperm(length(Emotion_label));
train_idx = round(0.90*size(Emotion_label, 1));
% Performing Training on 90% of the dataset and Testing on 10% of the
% dataset
feature_vect_train = featureVectLandmrkHaarEigen(idx(1:train_idx),:);
class_train = Emotion_label(idx(1:train_idx),:);
SVMMulticlass_model_c = fitcecoc(feature_vect_train, class_train, 'Coding', 'onevsall',...
                                    'Learners', 'svm'); 
trainLabel_emotion = predict(SVMMulticlass_model_c, feature_vect_train);

target_emotion = Emotion_label(idx(train_idx+1:end),:);
feature_vect_test = featureVectLandmrkHaarEigen(idx(train_idx+1:end),:);
predictLabel_emotion = predict(SVMMulticlass_model_c, feature_vect_test);

% Computing Training Error
train_loss = eval_mcr(trainLabel_emotion, class_train);
% Computing Testing Error
test_loss = eval_mcr(predictLabel_emotion, target_emotion);
%[conf_LHE, confmat_LHE, ind_LHE, error_LHE] = confusion(target_emotion, predictLabel_emotion);

% Performing 10 fold cross vaalidation
feature_combc = featureVectLandmrkHaarEigen(idx(1:end),:);
emotion_labels = Emotion_label(idx(1:end),:);
SVMMulticlass_c = fitcecoc(feature_combc, emotion_labels, 'Coding', 'onevsall',...
                                    'Learners', 'svm'); 

LandmarkHaarEigenCrossVal = crossval(SVMMulticlass_c, 'KFold', 10);
cross_val_combc_error = kfoldLoss(LandmarkHaarEigenCrossVal);
fprintf('The 10 fold cross-validation error using Landmark + Haar+ Eigenfaces features = %f\n', cross_val_combc_error);


%LandmarkHaarEigenfacesCombo = struct('conf_LHE',conf_LHE,'confmat_LHE', confmat_LHE, 'ind_LHE', ind_LHE, ...
%                             'error_LHE',error_LHE, 'cross_val_LHE', cross_val_combc_error, 'features', 'LandmarkHaarEigenfaces');
%save('LandmarkHaarEigenfacesClassifyResults.mat', 'LandmarkHaarEigenfacesCombo');                         
