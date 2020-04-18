function [Y_pred, A] = LSTM_predict(X, param, r_or_c)

% X: input data, shape: (n_input, m_test, t_time)
% Y: output data, shape: (n_output, m_test, t_time)
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

[~, m_test, ~] = size(X);
[A, Y_aux, ~] = LSTM_forward_prop(X, param, r_or_c);

if strcmp(r_or_c,'regression')
    Y_pred = Y_aux;
elseif strcmp(r_or_c,'classification')
    Y_pred = zeros(size(Y_aux));
    for i = 1:m_test
        idx = find(Y_aux(:,i)==max(Y_aux(:,i)));
        Y_pred(idx,i) = 1;
    end
end

end