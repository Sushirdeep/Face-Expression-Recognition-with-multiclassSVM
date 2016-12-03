% Program to Compute the Eigen Faces

clear; clc; 
load('FeatureFiles\FaceLabelData.mat');
num_images= size(Face_labeldata,1);
Face_labeldata = Face_labeldata' ;

% Finding the mean image and the mean-shifted input images
mean_face = mean(Face_labeldata, 2);
shifted_Faceimages = Face_labeldata - repmat(mean_face, 1, num_images);

% Calculating the ordered eigenvectors and eigenvalues
[evectors, score, evalues]  = pca(Face_labeldata');
% Only retaining the top 'num_eigenfaces' eigenvectors (i.e. the principal components)
num_eigenfaces = 200;
evectors = evectors(:, 1:num_eigenfaces);

% Projecting the images into the subspace to generate the feature vectors
featureVectEigenFaces = evectors' * shifted_Faceimages;

%Saving EigenFaces Features
save('FeatureFiles\EigenFaceFeaturesFaceExp.mat', 'featureVectEigenFaces');