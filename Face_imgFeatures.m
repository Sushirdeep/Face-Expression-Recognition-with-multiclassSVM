% Reading the Relevant Face Images
clear; clc;
load('EmotionsLabels.mat');
% Loading the Model of the Face Detection from the Viola Jones Algorithm
faceDetector = vision.CascadeObjectDetector;


% Identify the Emotion Coded Files
addpath('Emotion');
addpath('cohn-kanade-images');
addpath('Landmarks');
Face_labeldata= zeros(size(Emotion_seq,1), 3600);

for  i=1:size(Emotion_seq, 1)
  %% Detecting the Faces in images
 
  Filepath =  char(Emotion_seq(i,:));
  Filepath_imgs= strcat('cohn-kanade-images\', Filepath);
  ImgSequenceList= dir(Filepath_imgs);
  j= size(ImgSequenceList,1);
  ImgEmotion_filepath = strcat(Filepath_imgs, '\', ImgSequenceList(j).name);
  Img= imread(ImgEmotion_filepath);
  
  Facebboxes = step(faceDetector, Img);
  if size(Facebboxes,1)>1
      for k=1:size(Facebboxes,1)
          Area_face(k)= Facebboxes(k,3)*Facebboxes(k,4);
      end
      [value, idx] = max(Area_face);
      Facebox= Facebboxes(k,:);
  else
      Facebox= Facebboxes;
  end
  
  Face_firstpt_x = Facebox(1,1);
  Face_firstpt_y = Facebox(1,2);
  if Face_firstpt_y+ Facebox(1,3)+ 20 < size(Img,1)
       Face_width= Facebox(1,3)+ 20;
  else 
      Face_width = Facebox(1,3);
  end
  
  if Face_firstpt_x+ Facebox(1,4)+ 20 < size(Img,2)
      Face_height= Facebox(1,4)+20;
  else
      Face_height= Facebox(1,4);
  end
  
  IFaces = insertObjectAnnotation(Img, 'rectangle', Facebox, 'Face');
  % figure, imshow(Img), title('Detected faces');
  Face_img =zeros(Face_height, Face_width);
  img_face= double(Img);
  for x=1: Face_height
      for y=1: Face_width
          Face_img(y,x)= img_face((Face_firstpt_y+ y)- 1, (Face_firstpt_x+ x) -1 );
      end
  end
  
  % Face Image Extracted from the database
  % figure, imshow(Face_img,[]);
 
  % Resizing the images to a standard size for Gabor and Haar Features
  Face_img1 = imresize(Face_img, [128, 128]);
  Face_img1 =uint8(Face_img1);
  Face_img1= double(Face_img1);
  % Resizing the image to a small standard size for eigenfaces
  Face_img2 = imresize(Face_img, [60, 60]);
  Face_img2= uint8(Face_img2);
  Face_img2 = (Face_img2(:)') ;
  Face_labeldata(i,:) =  Face_img2 ; 
  

   %% Extarcting the Gabor Features
    Scales_val=5;
    Orient_val=8;
    Grid_x= 39;
    Grid_y= 39;
    gaborArray= gaborFilterBank(Scales_val, Orient_val, Grid_x, Grid_y);
    down_samp_x= 4;
    down_samp_y= 4;
    featureVectGabor(:,i)= gaborFeatures(Face_img1, gaborArray, down_samp_x, down_samp_y);
     
    
    %% Extracting the Haar Wavelet Features
    
    N=4; % Order of the wavelets
    wname= 'haar';  % Type of Wavelets
    [Coeff, Scal]= wavedec2(Face_img1, N, wname);
    Coeff= Coeff';
    featureVectHaarWav(:,i)= Coeff; 
    
    
    
    %% Extracting features using Morphological Operations
    %  Using Intensities for segmenting the image  
    
    img_gray = uint8(Face_img); 
    img_gray= histeq(img_gray);
    [thresh_level, effectiveness]= graythresh(img_gray);
    binary_imgFace= im2bw(img_gray, thresh_level);
    
   % Creating a Stuctural Element of size 3x3 for eroision
    se= strel('square', 3);
    img_Face_eroded= imerode(binary_imgFace, se);
    
    img_Faceboundary= binary_imgFace-img_Face_eroded;
    % figure, imshow(img_Faceboundary);
    Faceboundary = imresize(img_Faceboundary,[80 80]);
    %figure, imshow(Faceboundary);
    featureVectMorphBoundary(:,i)= Faceboundary(:); 
    
    
    
    
    %% Reading the Landmark Points
     folder_name= strcat('Landmarks\', Filepath); 
     LandmarksFileList= dir(folder_name);
      j= size(LandmarksFileList,1);
      Landmark_filepath = strcat(folder_name, '\', LandmarksFileList(j).name);
      Landmark_pts= [];
      fp= fopen(Landmark_filepath, 'r');
      
      readLine= fgetl(fp);
      while(ischar(readLine))
          LineContents = regexp(readLine, '\s+', 'split');
          Landmark_pts= [Landmark_pts; str2double(LineContents)];
          readLine= fgetl(fp);
      end
      
      
      
      
      fclose(fp);
    % Extracting the Landmark Feature Points  
    Landmark_pts = Landmark_pts(:, 2:3);
    LandmarkFaceX = Landmark_pts(:,1) - Face_firstpt_x ;
    LandmarkFaceY = Landmark_pts(:,2) - Face_firstpt_y ;
    LandmarkFeaturePts = [ LandmarkFaceX(:) ; LandmarkFaceY(:) ];
    featureVectLandmarkPts(:, i) = LandmarkFeaturePts ;  
   

end

%% Saving the Feature Files as mat files
%  Saving the Gabor Features
save('FeatureFiles\GaborFeatureFaceExp.mat', 'featureVectGabor');

% Saving the Haar Wavelet Features
save('FeatureFiles\HaarWaveletFeatureFaceExp.mat','featureVectHaarWav');

% Saving the Face Image Labelled data for eigenfaces
save('FeatureFiles\FaceLabelData.mat', 'Face_labeldata');

% Saving Features obtained from Morphological Operations
save('FeatureFiles\MorphologicalFeatureFaceExp.mat', 'featureVectMorphBoundary');

%Saving the Landmark Features points
save('FeatureFiles\LandmarkPtsFaceExp.mat','featureVectLandmarkPts');

