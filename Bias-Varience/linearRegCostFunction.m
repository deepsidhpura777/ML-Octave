function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples
n=size(X,2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


temp = theta;
temp(1) = 0;
h=X *theta;

J = 1/(2*m)*sum((h-y).^2)+(lambda/(2*m))*(sum(temp .^2));

h = h * ones(1,n);
y = y * ones(1,n);

grad = (1/m) * sum((h-y) .* X)' + (lambda / m) * temp ;





% =========================================================================

grad = grad(:);

end
