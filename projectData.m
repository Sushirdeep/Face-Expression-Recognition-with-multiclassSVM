function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. 


Z = zeros(size(X, 1), K);


% Obtaining the top k eigenvectors of U (first k columns)
%  For the i-th example X(i,:), the projection on to the k-th 
%  eigenvector is given as follows:
%  x = X(i, :)';
%  projection_k = x' * U(:, k);

U_reduce=U(:,(1:K));
for i=1:size(X,1)
    x=X(i,:)';
    projection_k=x'*U_reduce;
    Z(i,:)=projection_k;
end




end
