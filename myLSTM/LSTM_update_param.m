function [param, v, s] = LSTM_update_param(param, grad_hidden, ...
    grad_output, v, s, beta_1, beta_2, t, epsilon, learning_rate, ...
    optimization)

% param: struct containing the following:
    % 1) W_f: weight matrix of forget gate, shape: (n_hidden, n_hidden + n_input)
    % 2) b_f: bias of the forget gate, shape: (n_hidden, 1)
    % 3) W_i: weight matrix of the update gate, shape: (n_hidden, n_hidden + n_input)
    % 4) b_i: bias of the update gate, shape: (n_hidden, 1)
    % 5) W_c: weigth matrix of the first "tanh", shape: (n_hidden, n_hidden + n_input)
    % 6) b_c: bias of the first "tanh", shape: (n_hidden, 1)
    % 7) W_o: weight matrix of the output gate, shape: (n_hidden, n_hidden + n_input)
    % 8) b_o: bias of the output gate, shape: (n_hidden, 1)
    % 9) W_y: weigth matrix relating hidden state to output, shape: (n_output, n_hidden)
    % 10) b_y: bias relating hidden state to output, shape: (n_output, 1)
    % 11) G: weighted connections matrix
% grad_hidden: struct containing the following:
    % 1) dW_f
    % 2) dW_i
    % 3) dW_c
    % 4) dW_o
    % 5) db_f
    % 6) db_i
    % 7) db_c
    % 8) db_o
% grad_output: struct containing the following:
    % 1) dW_y
    % 2) db_y
% v: parameter for update of Ws and bs ('adam' and 'momentum')
% s: parameter for update of Ws and bs ('adam')
% beta_1: update parameter of v (recommended: 0.9) ('adam' and 'momentum')
% beta_2: update parameter of s (recommended: 0.999) ('adam')
% t: counter for correction of parameters v and s ('adam')
% epsilon: parameter to prevent division by zero (recommended: 1e-8) ('adam')
% learning_rate: update parameter of parameters (recommended: 0.001)
% optimization: optimization method ('adam' or 'momentum')

if strcmp(optimization,'adam')
    
    v.dW_f = beta_1*v.dW_f + (1-beta_1)*grad_hidden.dW_f/(1-beta_1^t);
    v.dW_i = beta_1*v.dW_i + (1-beta_1)*grad_hidden.dW_i/(1-beta_1^t);
    v.dW_c = beta_1*v.dW_c + (1-beta_1)*grad_hidden.dW_c/(1-beta_1^t);
    v.dW_o = beta_1*v.dW_o + (1-beta_1)*grad_hidden.dW_o/(1-beta_1^t);
    v.dW_y = beta_1*v.dW_y + (1-beta_1)*grad_output.dW_y/(1-beta_1^t);
    v.db_f = beta_1*v.db_f + (1-beta_1)*grad_hidden.db_f/(1-beta_1^t);
    v.db_i = beta_1*v.db_i + (1-beta_1)*grad_hidden.db_i/(1-beta_1^t);
    v.db_c = beta_1*v.db_c + (1-beta_1)*grad_hidden.db_c/(1-beta_1^t);
    v.db_o = beta_1*v.db_o + (1-beta_1)*grad_hidden.db_o/(1-beta_1^t);
    v.db_y = beta_1*v.db_y + (1-beta_1)*grad_output.db_y/(1-beta_1^t);

    s.dW_f = beta_2*s.dW_f + (1-beta_2)*grad_hidden.dW_f.* ...
        grad_hidden.dW_f/(1-beta_2^t);
    s.dW_i = beta_2*s.dW_i + (1-beta_2)*grad_hidden.dW_i.* ...
        grad_hidden.dW_i/(1-beta_2^t);
    s.dW_c = beta_2*s.dW_c + (1-beta_2)*grad_hidden.dW_c.* ...
        grad_hidden.dW_c/(1-beta_2^t);
    s.dW_o = beta_2*s.dW_o + (1-beta_2)*grad_hidden.dW_o.* ...
        grad_hidden.dW_o/(1-beta_2^t);
    s.dW_y = beta_2*s.dW_y + (1-beta_2)*grad_output.dW_y.* ...
        grad_output.dW_y/(1-beta_2^t);
    s.db_f = beta_2*s.db_f + (1-beta_2)*grad_hidden.db_f.* ...
        grad_hidden.db_f/(1-beta_2^t);
    s.db_i = beta_2*s.db_i + (1-beta_2)*grad_hidden.db_i.* ...
        grad_hidden.db_i/(1-beta_2^t);
    s.db_c = beta_2*s.db_c + (1-beta_2)*grad_hidden.db_c.* ...
        grad_hidden.db_c/(1-beta_2^t);
    s.db_o = beta_2*s.db_o + (1-beta_2)*grad_hidden.db_o.* ...
        grad_hidden.db_o/(1-beta_2^t);
    s.db_y = beta_2*s.db_y + (1-beta_2)*grad_output.db_y.* ...
        grad_output.db_y/(1-beta_2^t);

    param.W_f = param.W_f - learning_rate*v.dW_f./(sqrt(s.dW_f)+epsilon);
    param.W_i = param.W_i - learning_rate*v.dW_i./(sqrt(s.dW_i)+epsilon);
    param.W_c = param.W_c - learning_rate*v.dW_c./(sqrt(s.dW_c)+epsilon);
    param.W_o = param.W_o - learning_rate*v.dW_o./(sqrt(s.dW_o)+epsilon);
    param.W_y = param.W_y - learning_rate*v.dW_y./(sqrt(s.dW_y)+epsilon);
    param.b_f = param.b_f - learning_rate*v.db_f./(sqrt(s.db_f)+epsilon);
    param.b_i = param.b_i - learning_rate*v.db_i./(sqrt(s.db_i)+epsilon);
    param.b_c = param.b_c - learning_rate*v.db_c./(sqrt(s.db_c)+epsilon);
    param.b_o = param.b_o - learning_rate*v.db_o./(sqrt(s.db_o)+epsilon);
    param.b_y = param.b_y - learning_rate*v.db_y./(sqrt(s.db_y)+epsilon);
    
elseif strcmp(optimization,'momentum')
    
    v.dW_f = beta_1*v.dW_f + (1-beta_1)*grad_hidden.dW_f;
    v.dW_i = beta_1*v.dW_i + (1-beta_1)*grad_hidden.dW_i;
    v.dW_c = beta_1*v.dW_c + (1-beta_1)*grad_hidden.dW_c;
    v.dW_o = beta_1*v.dW_o + (1-beta_1)*grad_hidden.dW_o;
    v.dW_y = beta_1*v.dW_y + (1-beta_1)*grad_output.dW_y;
    v.db_f = beta_1*v.db_f + (1-beta_1)*grad_hidden.db_f;
    v.db_i = beta_1*v.db_i + (1-beta_1)*grad_hidden.db_i;
    v.db_c = beta_1*v.db_c + (1-beta_1)*grad_hidden.db_c;
    v.db_o = beta_1*v.db_o + (1-beta_1)*grad_hidden.db_o;
    v.db_y = beta_1*v.db_y + (1-beta_1)*grad_output.db_y;

    s = 0;

    param.W_f = param.W_f - learning_rate*v.dW_f;
    param.W_i = param.W_i - learning_rate*v.dW_i;
    param.W_c = param.W_c - learning_rate*v.dW_c;
    param.W_o = param.W_o - learning_rate*v.dW_o;
    param.W_y = param.W_y - learning_rate*v.dW_y;
    param.b_f = param.b_f - learning_rate*v.db_f;
    param.b_i = param.b_i - learning_rate*v.db_i;
    param.b_c = param.b_c - learning_rate*v.db_c;
    param.b_o = param.b_o - learning_rate*v.db_o;
    param.b_y = param.b_y - learning_rate*v.db_y;

end

end