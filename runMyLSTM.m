function [final_param, cost_train, subsets, min_length] = ...
    runMyLSTM(data,param_net)

% param_net:
    % .dim_reduce_method = 'fa'; % 'none', 'pca', 'fa', 'ppca', 'isomap'
        % If dim_reduce_method = 'isomap', data = isomap_data (this is because isomap is time consuming -> done outside the function)
    % .dims = 5;
    % .percentage_train = 0.8;
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

dim_reduce_method = param_net.dim_reduce_method;
if strcmp(dim_reduce_method,'none')
    dim_reduce_method = 'spikes';
    dims = length(data(1).(strcat('M1_',dim_reduce_method))(1,:));
    td = data;
elseif strcmp(dim_reduce_method,'isomap')
    dims = param_net.dims;
    td = data;
else
    dims = param_net.dims;
    clear param_dim
    param_dim.algorithm = dim_reduce_method;
    param_dim.signals = 'M1_spikes';
    param_dim.use_trials = 1:length(data);
    param_dim.num_dims = dims;
    [td,~] = dimReduce(data,param_dim);
end

percentage_train = param_net.percentage_train;
m_train = fix(length(td)*percentage_train);

min_length = 100;
for i = 1:length(td)
    if min_length > size(td(i).M1_spikes,1)
        min_length = size(td(i).M1_spikes,1);
    end
end

% Train and test sets:

idx = randperm(length(td));

x = zeros(dims,length(td),min_length);
for i = 1:dims
    for j = 1:length(td)
        for k = 1:min_length
            if strcmp(dim_reduce_method,'none')
                x(i,j,k) = td(idx(j)).M1_spikes(k,i);
            else
                x(i,j,k) = td(idx(j)).(strcat('M1_',dim_reduce_method))(k,i);
            end
        end
    end
end

y = zeros(2,length(td),min_length);
for i = 1:2
    for j = 1:length(td)
        for k = 1:min_length
            y(i,j,k) = td(idx(j)).vel(k,i);
        end
    end
end

x_train = x(:,1:m_train,:);
y_train = y(:,1:m_train,:);
x_test = x(:,m_train+1:end,:);
y_test = y(:,m_train+1:end,:);

%{
m_x_train = mean(x_train,2:3);
s_x_train = std(x_train,0,2:3);
x_train = (x_train-m_x_train)./s_x_train;
m_y_train = mean(y_train,2:3);
s_y_train = std(y_train,0,2:3);
y_train = (y_train-m_y_train)./s_y_train;
m_x_test = mean(x_test,2:3);
s_x_test = std(x_test,0,2:3);
x_test = (x_test-m_x_test)./s_x_test;
m_y_test = mean(y_test,2:3);
s_y_test = std(y_test,0,2:3);
y_test = (y_test-m_y_test)./s_y_test;
%}

% Training:

[final_param, cost_train] = LSTM_train(x_train, y_train, ...
    param_net.mini_batch_size, param_net.num_epochs, ...
    param_net.n_hidden, param_net.beta_1, param_net.beta_2, ...
    param_net.epsilon, param_net.learning_rate, ...
    param_net.optimization, param_net.transfer_learning, ...
    param_net.transfer_param, param_net.r_or_c);

% Subsets:

subsets.train.X = x_train;
subsets.train.Y = y_train;
subsets.test.X = x_test;
subsets.test.Y = y_test;

end