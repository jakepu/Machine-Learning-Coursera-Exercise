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

hidden_layer = [ones(size(X,1),1),X] * Theta1';
hidden_layer_z = hidden_layer;
hidden_layer = sigmoid(hidden_layer);
output_layer = [ones(size(hidden_layer,1),1),hidden_layer] * Theta2';
%[~,p] = max(output_layer,[],2);
for i = 1:m
    %h = zeros(10,1);
    %h(p(i)) = 1;
    h = sigmoid(output_layer(i,:)');
    log_h = log(h);
    y_i = zeros(num_labels,1);
    y_i(y(i)) = 1;
    J = J + sum(y_i .* log_h + (1 - y_i) .* log(1 - h));
end
J = -1/m * J;
extra_J = lambda/2/m * (sum(Theta1(:,2:end) .^ 2,'all')+sum(Theta2(:,2:end) .^ 2,'all'));
J = J + extra_J;


triangle_1 = zeros(size(Theta1));
triangle_2 = zeros(size(Theta2));
for i = 1:m
    y_i = zeros(num_labels,1);
    y_i(y(i)) = 1;
    delta_3 = sigmoid(output_layer(i,:)') - y_i;
    delta_2 = Theta2' * delta_3 .* [0;sigmoidGradient(hidden_layer_z(i,:)')];
    delta_2 = delta_2(2:end);
    triangle_1 = triangle_1 + delta_2 * [1,X(i,:)];
    triangle_2 = triangle_2 + delta_3 *[1,hidden_layer(i,:)];
end
grad = 1/m * [triangle_1(:);triangle_2(:)];

extra_theta1 = lambda / m * Theta1;
extra_theta1(:,1) = zeros(size(extra_theta1,1),1);
extra_theta2 = lambda / m * Theta2;
extra_theta2(:,1) = zeros(size(extra_theta2,1),1);
grad = grad + [extra_theta1(:);extra_theta2(:)];








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
