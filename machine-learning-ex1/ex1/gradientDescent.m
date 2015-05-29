function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
x1=zeros(m,1);
x2=zeros(m,1);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


%x1=X(:,1);
%x2=X(:,2);

%x(n)=X(:,n);

%theta(1)= theta(1)- alpha*1/m* (x1'* (X*theta-y));
%theta(2)= theta(2)- alpha*1/m* (x2'* (X*theta-y));

%vectorization
theta = theta - (alpha/m) * (X' * (X*theta-y));
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
