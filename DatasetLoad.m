% Loading the Cohn Kanade Datset

clc; clear; close all;
% Identify the Emotion Coded Files
addpath('Emotion');
addpath('cohn-kanade-images');
addpath('Landmarks');

Emotion_seq = [];
EmotionList= dir('Emotion');
k=1;

for i=3:size(EmotionList,1)
    EmotionListPath = EmotionList(i).name;
    if (isdir(strcat('Emotion','\',EmotionListPath)))
        EmotionSeqList = dir(strcat('Emotion', '\', EmotionListPath));
        for j=3: size(EmotionSeqList,1)
            EmotionSeq= dir(strcat('Emotion\',EmotionListPath, '\',(EmotionSeqList(j).name)));
            if size(EmotionSeq,1)==3
                EmotionPath= strcat('Emotion\',EmotionListPath, '\',(EmotionSeqList(j).name));
                Filename= strcat(EmotionPath, '\', EmotionSeq(3).name);
               
                  fid = fopen(Filename);
                  tline = fgetl(fid);
                  fclose(fid);
                Emotion_label(k)= str2num(tline);
                Emotion_seq(k,:) = strcat(EmotionListPath, '\', EmotionSeqList(j).name);
                
                k= k+1 ;
            end    
        end
    end
end

            
 


Emotion_label = Emotion_label';
save('EmotionsLabels.mat', 'Emotion_label', 'Emotion_seq');