function [param, cost_train] = LSTM_train(X, Y, mini_batch_size, ...
    num_epochs, n_hidden, beta_1, beta_2, epsilon, learning_rate, ...
    optimization, transfer_learning, transfer_param, r_or_c, lambda)

% X: input data, shape: (n_input, m_trials, t_time)
% Y: output data, shape: (n_output, m_trials, t_time)
% mini_batch_size: desired mini batch size
% num_epochs: desired number of epochs
% n_hidden: number of hidden units
% beta_1: update parameter of v (recommended: 0.9) ('adam' and 'momentum')
% beta_2: update parameter of s (recommended: 0.999) ('adam')
% epsilon: parameter to prevent division by zero (recommended: 1e-8) ('adam')
% learning_rate: update parameter of parameters (recommended: 0.001)
% optimization: optimization method ('adam' or 'momentum')
% transfer_learning: 'true' or 'false'
% transfer_param: transfered parameters
% r_or_c: 'regression' or 'classification'

[n_input, m_trials, t_time] = size(X);
[n_output, ~, ~] = size(Y);

if strcmp(transfer_learning,'true')
    param = transfer_param;
    [~, v, s] = LSTM_initialization(n_input, n_hidden, n_output);
else
    [param, v, s] = LSTM_initialization(n_input, n_hidden, n_output);
end

num_batch = fix(m_trials/mini_batch_size);
t = 0;
costs = [];

for i = 1:num_epochs
    idx = randperm(m_trials);
    costs_epoch = [];

    for j = 1:num_batch
        mini_batch_X = zeros(n_input, mini_batch_size, t_time);
        if strcmp(r_or_c,'regression')
            mini_batch_Y = zeros(n_output, mini_batch_size, t_time);
        elseif strcmp(r_or_c,'classification')
            mini_batch_Y = zeros(n_output, mini_batch_size);
        end

        for k = 1:mini_batch_size
            mini_batch_X(:,k,:) = X(:,idx((j-1)*mini_batch_size+k),:);
            if strcmp(r_or_c,'regression')
                mini_batch_Y(:,k,:) = Y(:,idx((j-1)*mini_batch_size+k),:);
            elseif strcmp(r_or_c,'classification')
                mini_batch_Y(:,k) = Y(:,idx((j-1)*mini_batch_size+k));
            end
        end

        [A, Y_pred, cache] = LSTM_forward_prop(mini_batch_X, param, ...
            r_or_c);
        [grad_output, dA, cost_mini_batch] = LSTM_cost(mini_batch_Y, ...
            Y_pred, A, param, r_or_c, lambda);
        grad_hidden = LSTM_backward_prop(mini_batch_X, dA, cache, ...
            param, r_or_c);
        t = t + 1;
        [param, v, s] = LSTM_update_param(param, grad_hidden, ...
            grad_output, v, s, beta_1, beta_2, t, epsilon, ...
            learning_rate, optimization);

        costs_epoch = [costs_epoch cost_mini_batch];
    end

    costs = [costs mean(costs_epoch)];
    figure(1);
    aux = linspace(1,length(costs),length(costs));
    plot(aux,costs);
    title('COST');
    ylabel('cost');
    xlabel('epoch');
end

cost_train = costs(end);

end