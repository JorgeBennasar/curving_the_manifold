function [param, v, s] = LSTM_initialization(n_input, n_hidden, n_output)

% n_input: size of input data
% n_hidden: number of hidden units
% n_output: size of output data

param.W_f = randn(n_hidden, n_hidden+n_input)*sqrt(2/n_input);
param.W_i = randn(n_hidden, n_hidden+n_input)*sqrt(2/n_input);
param.W_c = randn(n_hidden, n_hidden+n_input)*sqrt(2/n_input);
param.W_o = randn(n_hidden, n_hidden+n_input)*sqrt(2/n_input);
param.W_y = randn(n_output, n_hidden)*sqrt(2/n_hidden);
param.b_f = zeros(n_hidden, 1);
param.b_i = zeros(n_hidden, 1);
param.b_c = zeros(n_hidden, 1);
param.b_o = zeros(n_hidden, 1);
param.b_y = zeros(n_output, 1);

v.dW_f = zeros(size(param.W_f));
v.dW_i = zeros(size(param.W_i));
v.dW_c = zeros(size(param.W_c));
v.dW_o = zeros(size(param.W_o));
v.dW_y = zeros(size(param.W_y));
v.db_f = zeros(size(param.b_f));
v.db_i = zeros(size(param.b_i));
v.db_c = zeros(size(param.b_c));
v.db_o = zeros(size(param.b_o));
v.db_y = zeros(size(param.b_y));

s.dW_f = zeros(size(param.W_f));
s.dW_i = zeros(size(param.W_i));
s.dW_c = zeros(size(param.W_c));
s.dW_o = zeros(size(param.W_o));
s.dW_y = zeros(size(param.W_y));
s.db_f = zeros(size(param.b_f));
s.db_i = zeros(size(param.b_i));
s.db_c = zeros(size(param.b_c));
s.db_o = zeros(size(param.b_o));
s.db_y = zeros(size(param.b_y));

end