function loss = eval_mcr(predictions,targets)
% Evaluate the mis-classification rate loss
% Loss = eval_mcr(Predictions,Targets)
%
% In:
%   Predictions : vector of predictions made by the classifier
%
%   Targets : vector of true labels (-1 / +1)
%
% Out:
%   Loss : mis-classification rate
predictions = predictions(:);
targets = targets(:) ;
loss = mean(predictions~=targets);
end