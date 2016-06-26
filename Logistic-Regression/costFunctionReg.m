function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n=size(X,2);
h=1./(1+ e.^-(X*theta));

J=1/m*sum(-y.*log(h)-(1-y).*log(1-h))+(lambda/(2*m))*(sum(theta.^2)-(theta(1).^2));

h=h * ones(1,n);
y=y * ones(1,n);

grad=1/m .* sum((h-y).* X)' ;

for j=2:n

grad(j)=grad(j)+(lambda/m)*(theta(j));

end




% =============================================================

end
