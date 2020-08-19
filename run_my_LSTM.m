function [final_param, cost_train, subsets] = ...
    run_my_LSTM(data, param_net)

% param_net:
    % .m_train_per_target = 30; number of training trials per target
    % .mini_batch_size = 24;
    % .num_epochs = 100;
    % .stop_condition = 5; if 0, no stop condition
    % .n_hidden = 3;
    % .beta_1 = 0.9;
    % .beta_2 = 0.999;
    % .epsilon = 1e-8;
    % .learning_rate = 0.005;
    % .learning_rate_change = 'yes' or 'no'
    % .learning_rate_rule = 1/2; learning_rate_now = learning_rate/i^(learning_rate_rule)
    % .optimization = 'adam' or 'momentum'
    % .transfer_learning = 'true' or 'false'
    % .transfer_param (if transfer_learning = 'false', stays unused)
    % .r_or_c = 'regression' or 'classification'
    % .lambda = 0
    % .stop_condition: cost stop threshold (0 for no threshold)
    % .learning_rate_change = 'yes' or 'no'
    % .learning_rate_rule = 1/3 (learning_rate_now = learning_rate/i^(learning_rate_rule), i being the epoch)
    % .connectivity = percentage of connectivity between neurons (0 to 1)
    % .mode = mode of training (1 or 2)

% Initialization:

if param_net.mode == 1
    m_train_per_target = param_net.m_train_per_target;
    m_train = length(data.target_sel)*m_train_per_target;
    time = size(data.X,3);
    x_train = zeros(size(data.X,1),m_train,time);
    y_train = zeros(2,m_train,time);
    index_train = zeros(1,m_train);
    x_test = zeros(size(data.X,1),size(data.X,2)-m_train,time);
    y_test = zeros(2,size(data.X,2)-m_train,time);
    index_test = zeros(1,size(data.X,2)-m_train);
    target_counter = zeros(1,8);
    counter_train = 0;
    counter_test = 0;
    for i = 1:size(data.X,2)
        for j = data.target_sel
            if j == data.targets(i) 
                if target_counter(j) < m_train_per_target
                    target_counter(j) = target_counter(j) + 1;
                    counter_train = counter_train + 1;
                    x_train(:,counter_train,:) = data.X(:,i,:);
                    for k = 1:2
                        for l = 1:time
                            y_train(k,counter_train,l) = data.Y(k,i,l);
                        end
                    end
                    index_train(counter_train) = i;
                else
                    counter_test = counter_test + 1;
                    x_test(:,counter_test,:) = data.X(:,i,:);
                    for k = 1:2
                        for l = 1:time
                            y_test(k,counter_test,l) = data.Y(k,i,l);
                        end
                    end
                    index_test(counter_test) = i;
                end
            end
        end
    end
    idx_train = randperm(length(index_train));
    index_train = index_train(idx_train);
    x_train = x_train(:,idx_train,:);
    y_train = y_train(:,idx_train,:);
    idx_test = randperm(length(index_test));
    index_test = index_test(idx_test);
    x_test = x_test(:,idx_test,:);
    y_test = y_test(:,idx_test,:);
elseif param_net.mode == 2
    x_train = data.x_train;
    y_train = data.y_train;
end

% Training:

[final_param, cost_train] = LSTM_train(x_train, y_train, ...
    param_net.mini_batch_size, param_net.num_epochs, ...
    param_net.n_hidden, param_net.beta_1, param_net.beta_2, ...
    param_net.epsilon, param_net.learning_rate, ...
    param_net.optimization, param_net.transfer_learning, ...
    param_net.transfer_param, param_net.r_or_c, param_net.lambda, ...
    param_net.stop_condition, param_net.learning_rate_change, ...
    param_net.learning_rate_rule, param_net.connectivity, ...
    param_net.amp_noise, param_net.noise_samples, ...
    param_net.correlation_reg);

% Subsets:

if param_net.mode == 1
    subsets.train.X = x_train;
    subsets.train.Y = y_train;
    subsets.train.index = index_train;
    subsets.test.X = x_test;
    subsets.test.Y = y_test;
    subsets.test.index = index_test;
elseif param_net.mode == 2
    subsets = [];
end

end