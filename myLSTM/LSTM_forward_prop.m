function [A, Y_pred, cache] = LSTM_forward_prop(X, param, r_or_c)

% X: input data, shape: (n_input, m_trials, t_time)
% param: struct containing the following:
    % 1) W_f: weight matrix of forget gate, shape: (n_hidden, n_hidden + n_input)
    % 2) b_f: bias of the forget gate, shape: (n_hidden, 1)
    % 3) W_i: weight matrix of the update gate, shape: (n_hidden, n_hidden + n_input)
    % 4) b_i: bias of the update gate, shape: (n_hidden, 1)
    % 5) W_c: weigth matrix of the first "tanh", shape: (n_hidden, n_hidden + n_input)
    % 6) b_c: bias of the first "tanh", shape: (n_hidden, 1)
    % 7) W_o: weight matrix of the output gate, shape: (n_hidden, n_hidden + n_input)
    % 8) b_o: bias of the output gate, shape: (n_hidden, 1)
    % 5) W_y: weigth matrix relating hidden state to output, shape: (n_output, n_hidden)
    % 5) b_y: bias relating hidden state to output, shape: (n_output, 1)
% r_or_c: 'regression' or 'classification'

[~, m_trials, t_time] = size(X);
[n_output, n_hidden] = size(param.W_y);

A = zeros(n_hidden, m_trials, t_time);
C = zeros(n_hidden, m_trials, t_time);

A_next = zeros(n_hidden, m_trials);
C_next = zeros(n_hidden, m_trials);

if strcmp(r_or_c,'regression')
    Y_pred = zeros(n_output, m_trials, t_time);
end

for t = 1:t_time
    A_prev = A_next;
    C_prev = C_next;
    X_t = X(:,:,t); 
    concat = [A_prev; X_t];
    Ft = 1./(1+exp(-param.W_f*concat + param.b_f));
    It = 1./(1+exp(-param.W_i*concat + param.b_i));
    Cct = tanh(param.W_c*concat + param.b_c);
    C_next = C_prev.*Ft + It.*Cct;
    C(:,:,t) = C_next;
    Ot = 1./(1+exp(-param.W_o*concat + param.b_o));
    A_next = Ot.*tanh(C_next);
    A(:,:,t) = A_next;
    if strcmp(r_or_c,'regression')
        Y_pred(:,:,t) = param.W_y*A_next + param.b_y;
    end
    cache(t).A_next = A_next;
    cache(t).A_prev = A_prev;
    cache(t).C_next = C_next;
    cache(t).C_prev = C_prev;
    cache(t).Ft = Ft;
    cache(t).It = It;
    cache(t).Cct = Cct;
    cache(t).Ot = Ot;
end
 
if strcmp(r_or_c,'classification')
    Z_pred = param.W_y*A_next + param.b_y;
    Y_pred = softmax(Z_pred);
end

end