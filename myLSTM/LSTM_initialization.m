function [param, v, s] = LSTM_initialization(n_input, n_hidden, ...
    n_output, connectivity)

% n_input: size of input data
% n_hidden: number of hidden units
% n_output: size of output data
% connectivity = percentage of connectivity between neurons (0 to 1)

G_1 = erdos_renyi(n_hidden, connectivity);
G_2 = ones(n_hidden, n_input);
G = [G_1 G_2];
param.G = G;

param.W_f = randn(n_hidden, n_hidden+n_input)*sqrt(2/(n_hidden+n_input)).*G; 
param.W_i = randn(n_hidden, n_hidden+n_input)*sqrt(2/(n_hidden+n_input)).*G;
param.W_c = randn(n_hidden, n_hidden+n_input)*sqrt(2/(n_hidden+n_input)).*G;
param.W_o = randn(n_hidden, n_hidden+n_input)*sqrt(2/(n_hidden+n_input)).*G; 
param.W_y = randn(n_output, n_hidden)*sqrt(2/n_hidden);
param.b_f = zeros(n_hidden, 1);
param.b_i = zeros(n_hidden, 1);
param.b_c = zeros(n_hidden, 1);
param.b_o = zeros(n_hidden, 1);
param.b_y = zeros(n_output, 1);

param.init.W_f = param.W_f;
param.init.W_i = param.W_i;
param.init.W_c = param.W_c;
param.init.W_o = param.W_o;
param.init.W_y = param.W_y;
param.init.b_f = param.b_f;
param.init.b_i = param.b_i;
param.init.b_c = param.b_c;
param.init.b_o = param.b_o;
param.init.b_y = param.b_y;

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