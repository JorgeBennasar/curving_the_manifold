function [grad_output, dA, cost_mini_batch] = LSTM_cost(Y, Y_pred, A, ...
    param, r_or_c, lambda)

% Y: output data, shape: (n_output, m_trials, t_time)
% Y_pred: predicted output data, shape: (n_output, m_trials, t_time)
% A: hidden states, shape: (n_hidden, m_trials, t_time)
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

[~, m_trials, t_time] = size(A);

dW_y = zeros(size(param.W_y));
db_y = zeros(size(param.b_y));

if strcmp(r_or_c,'regression')

    dA = zeros(size(A));
    cost_mini_batch = 1/2/t_time/m_trials*sum(sum(sum((Y_pred-Y).^2,1), ...
        2),3) + lambda/m_trials*sum(sum(transpose(param.W_y) ...
        *param.W_y,1),2);

    for t = 1:t_time
        dY_pred = Y_pred(:,:,t)-Y(:,:,t); 
        dA(:,:,t) = transpose(param.W_y)*dY_pred;
        dW_y = dW_y + 1/m_trials*(dY_pred*transpose(A(:,:,t)));
        db_y = db_y + 1/m_trials*sum(dY_pred,2);
    end
    
elseif strcmp(r_or_c,'classification')

    cost_mini_batch = -sum(sum(Y.*log(Y_pred),1),2);
    dY_pred = Y_pred - Y;
    dA = transpose(param.W_y)*dY_pred;
    dW_y = dW_y + 1/m_trials*(dY_pred*transpose(A(:,:,end))) + ...
        lambda/m_trials*param.W_y;
    db_y = db_y + 1/m_trials*sum(dY_pred,2);
    
end

grad_output.dW_y = dW_y;
grad_output.db_y = db_y;

end