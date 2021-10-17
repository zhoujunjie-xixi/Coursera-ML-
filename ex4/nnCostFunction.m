function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X_plus = [ones(size(X, 1), 1) X];

z2 = X_plus * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
for i=1:size(X,1)
    y_plus = zeros(num_labels, 1);
    y_plus(y(i)) = 1;
    J = J + (-1)*log(h(i, :))*y_plus - (log(1 - h(i, :))*(1-y_plus));
end
J = J / m;

Theta1_plus = Theta1(:, 2:size(Theta1, 2)).^2;
plus1 = sum(sum(Theta1_plus));
Theta2_plus = Theta2(:, 2:size(Theta2, 2)).^2;
plus2 = sum(sum(Theta2_plus));

J = J + lambda*(plus1+plus2)/(2*m);

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for i = 1:m
    y_plus = zeros(num_labels, 1);
    y_plus(y(i)) = 1;
    delta_3 = a3(i, :)' - y_plus;
    % delta_3 (10¡Á1Î¬)
    delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z2(i, :)');  %Theta2 (10¡Á26)
    % delta_2 (25¡Á1Î¬)
    % a2(i,:) 1¡Á26
    % X_plus(i,:) 1¡Á401
    Delta_2 = Delta_2 + delta_3 * a2(i, :);
    Delta_1 = Delta_1 + delta_2 * X_plus(i, :);
end
%Theta1 Delta_1 25¡Á401
%Theta2 Delta_2 10¡Á26

Theta1_grad = Delta_1 ./ m;
Theta2_grad = Delta_2 ./ m;


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
