% Combining the feature files into the proposed combinations
% Only the significant features are combined because if all the dimensions are considered 
%the dimension exceed  more than the computer can handle

clc; clear; close all;
addpath('FeatureFiles');

load('FeatureFiles\GaborFeatureFaceExp.mat');

%% Obtaining the most Significant features of Gabor Coefficients
K_Gabor= 300;
featureVectGabor=featureVectGabor';
% Gabor Features were already Normalized when extracting the features
[U_Gabor, Scale_Gabor, eValueGabor]= pca(featureVectGabor);
S_GaborfeatureVect= projectData(featureVectGabor, U_Gabor, K_Gabor);
save('FeatureFiles\SignificantGaborFeatures.mat','S_GaborfeatureVect');

%% Obtiaining the most Significant features of Haar Wavelets
clear
load('FeatureFiles\HaarWaveletFeatureFaceExp.mat');
K_Haar= 200;

featureVectHaarWav = featureVectHaarWav';
% Normalizing the Haar Wavelet Features
[featureNorm_HaarWav, mu_Haar, sigma_HaarWav ]= featureNormalize(featureVectHaarWav);

[U_HaarWav, Scale_Haar, eValueHaar]= pca(featureNorm_HaarWav);
S_HaarfeatureVect= projectData(featureNorm_HaarWav, U_HaarWav, K_Haar);
save('FeatureFiles\SignificantHaarFeatures.mat','S_HaarfeatureVect');

%% Obtiaining the most Significant Morphological Image features
clear
load('FeatureFiles\MorphologicalFeatureFaceExp.mat');
%Most Significant Morphological Boundary Features
K_Morph= 180;
featureVectMorphBoundary = featureVectMorphBoundary';
[U_Morph, Scales_Morph, eValueMorph]= pca(featureVectMorphBoundary);
S_MorphBfeatureVect= projectData(featureVectMorphBoundary, U_Morph, K_Morph);
save('FeatureFiles\SignificantMorphFeatures.mat','S_MorphBfeatureVect');

%% Combining the Feature Vectors
clear
load('FeatureFiles\EigenFaceFeaturesFaceExp.mat');
load('FeatureFiles\LandmarkPtsFaceExp.mat');
load('FeatureFiles\SignificantGaborFeatures.mat');
load('FeatureFiles\SignificantHaarFeatures.mat');
load('FeatureFiles\SignificantMorphFeatures.mat');

% Feature Combination (a) Haar+ Gabor+Morphological boundary

featureVectHaarGaborMorph= [S_GaborfeatureVect, S_HaarfeatureVect, S_MorphBfeatureVect];

% FeatureCombination (b) Eigenfaces + Gabor + Haar
featureVectEigenFaces = featureVectEigenFaces';
featureVectHaarGaborEigenface = [featureVectEigenFaces, S_GaborfeatureVect, S_HaarfeatureVect];


% Feature Combination (c) Haar+ Eigenfaces+ Landmark
featureVectLandmarkPts = featureVectLandmarkPts';
featureVectLandmrkHaarEigen = [featureVectLandmarkPts, featureVectEigenFaces, S_HaarfeatureVect];

save('FeatureFiles\CombinationFeaturesHaarGaborMorph.mat','featureVectHaarGaborMorph');
save('FeatureFiles\CombinationFeaturesHaarGaborEigen.mat','featureVectHaarGaborEigenface');
save('FeatureFiles\CombinationFeaturesLandmarkHaarEigen.mat','featureVectLandmrkHaarEigen');