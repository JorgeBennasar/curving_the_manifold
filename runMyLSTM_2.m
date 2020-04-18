function [final_param, cost_train, subsets] = ...
    runMyLSTM_2(data, param_net)

% param_net:
    % .m_train_per_target = 30; number of training trials per target
    % .mini_batch_size = 24;
    % .num_epochs = 100;
    % .n_hidden = 3;
    % .beta_1 = 0.9;
    % .beta_2 = 0.999;
    % .epsilon = 1e-8;
    % .learning_rate = 0.005;
    % .optimization = 'adam' or 'momentum'
    % .transfer_learning = 'true' or 'false'
    % .transfer_param (if transfer_learning = 'false', stays unused)
    % .r_or_c = 'regression' or 'classification'
    
% Initialization:

td = data;
m_train_per_target = param_net.m_train_per_target;
m_train = 8*m_train_per_target;
time = size(td(1).input,2);

% Train and test sets:

x_train = zeros(size(td(1).input,1),m_train,time);
y_train = zeros(2,m_train,time);
index_train = zeros(1,m_train);
x_test = zeros(size(td(1).input,1),length(td)-m_train,time);
y_test = zeros(2,length(td)-m_train,time);
index_test = zeros(1,length(td)-m_train);

idx = randperm(length(td));
target_counter = zeros(1,8);
counter_train = 0;
counter_test = 0;
for i = 1:length(td)
    for j = 1:8
        if j == td(idx(i)).target 
            if target_counter(j) < m_train_per_target
                target_counter(j) = target_counter(j) + 1;
                counter_train = counter_train + 1;
                x_train(:,counter_train,:) = td(idx(i)).input;
                for k = 1:2
                    for l = 1:time
                        y_train(k,counter_train,l) = td(idx(i)).vel(l,k);
                    end
                end
                index_train(counter_train) = idx(i);
            else
                counter_test = counter_test + 1;
                x_test(:,counter_test,:) = td(idx(i)).input;
                for k = 1:2
                    for l = 1:time
                        y_test(k,counter_test,l) = td(idx(i)).vel(l,k);
                    end
                end
                index_test(counter_test) = idx(i);
            end
        end
    end
end

% Training:

[final_param, cost_train] = LSTM_train(x_train, y_train, ...
    param_net.mini_batch_size, param_net.num_epochs, ...
    param_net.n_hidden, param_net.beta_1, param_net.beta_2, ...
    param_net.epsilon, param_net.learning_rate, ...
    param_net.optimization, param_net.transfer_learning, ...
    param_net.transfer_param, param_net.r_or_c, param_net.lambda);

% Subsets:

subsets.train.X = x_train;
subsets.train.Y = y_train;
subsets.train.index = index_train;
subsets.test.X = x_test;
subsets.test.Y = y_test;
subsets.test.index = index_test;

end