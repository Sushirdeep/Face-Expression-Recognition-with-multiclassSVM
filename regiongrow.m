function [g, NR, SI, TI]= regiongrow(f, S, T)
% This function performs the Region Growing Algorithm for Segmentation
%S = seed valie , can be a single scalar value or array same size as f
%T= Threshold parameter, can be global threshold, single value or array
% g = Result of Region Growing Algorithm
% NR is the number of regions
% SI is the final seed image used by the algorithm
% TI is the final image that staisfied the threshold test before processed for connectivity
f= tofloat(f);

%if S is scalar obtain the seed image

if numel(S) == 1
    SI = f == S;
    S1 =S;
else
    % S is an array
    SI = bwmorph(S, 'shrink', Inf);
    S1= f(SI);
end

TI= false(size(f));
for K = 1:length(S1)
    seedvalue = S1(K);
    S = abs(f - seedvalue) <= T;
    TI= TI|S;
end

% Using imreconstruct to obtain the regions corresponding to each seed in S

[g , NR] = bwlabel(imreconstruct(SI,TI));
