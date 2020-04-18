function [net,subsets] = runLSTM(data,param_net)

% param:
    % .predict = {'vel'}, {'pos'}, {'acc'}, {'pos','vel'}...
    % .dim_reduce_method = 'fa'; % 'none', 'pca', 'fa', 'ppca', 'isomap'
        % If dim_reduce_method = 'isomap', data = isomap_data (this is because isomap is time consuming -> done outside the function)
    % .dims = 5;
    % .percentage_train = 0.8;
    % .percentage_val = 0.1;
    % .mini_batch_size = 24;
    % .max_epochs = 100;
    % .num_hidden_layers = 3;
    % .num_hidden_units = [150 100 50]; % Vector of number of units of the hidden layers (same length as .num_hidden_layers)
    % .dropout = [0.5 0.4 0.3]; % Vector of dropout probabilities (same length as .num_hidden_layers)
    % .method = 'adam'; % 'sgdm', 'rmsprop', 'adam'
    % .sequence_length = 'longest' % 'longest' to pad to the right, 'shortest' to truncate to the right
    
% Initialization:

predict = param_net.predict;
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
percentage_val = param_net.percentage_val;
num_seq_train = fix(percentage_train*length(td));
num_seq_val = fix(percentage_val*length(td));
num_features = dims;
num_responses = 0;
for i = 1:length(predict)
    num_responses = num_responses + length(data(1).(predict{(i)})(1,:));
end

layers = [];
layers = [layers sequenceInputLayer(num_features)];
for i = 1:param_net.num_hidden_layers
    layers = [layers lstmLayer(param_net.num_hidden_units(i),...
        'OutputMode','sequence')];
    layers = [layers dropoutLayer(param_net.dropout(i))];
end
layers = [layers fullyConnectedLayer(num_responses)];
layers = [layers regressionLayer];

% Train, validation and test sets:

% rand_pos = randperm(length(td));

X_train = cell(num_seq_train,1);
Y_train = cell(num_seq_train,1);
for i = 1:num_seq_train
    X_train{(i)} = transpose(td(i).(strcat('M1_', ... % transpose(td(rand_pos(i)).(strcat('M1_', ...
        dim_reduce_method)));
    Y_aux = [];
    for j = 1:length(predict)
        Y_aux = cat(1,Y_aux,transpose(td(i).(predict{(j)}))); % cat(1,Y_aux,transpose(td(rand_pos(i)).(predict{(j)})));
    end
    Y_train{(i)} = Y_aux;
end

X_val = cell(num_seq_val,1);
Y_val = cell(num_seq_val,1);
for i = 1:num_seq_val
    X_val{(i)} = transpose(td(num_seq_train+i). ... % transpose(td(rand_pos(num_seq_train+i)). ...
        (strcat('M1_',dim_reduce_method)));
    Y_aux = [];
    for j = 1:length(predict)
        Y_aux = cat(1,Y_aux,transpose(td(num_seq_train+i). ... % cat(1,Y_aux,transpose(td(rand_pos(num_seq_train+i)). ...
            (predict{(j)})));
    end
    Y_val{(i)} = Y_aux;
end

num_seq_test = length(td)-num_seq_train-num_seq_val;
X_test = cell(num_seq_test,1);
Y_test = cell(num_seq_test,1);
for i = 1:num_seq_test
    X_test{(i)} = transpose(td(num_seq_train+num_seq_val+i). ... % transpose(td(rand_pos(num_seq_train+num_seq_val+i)). ...
        (strcat('M1_',dim_reduce_method)));
    Y_aux = [];
    for j = 1:length(predict)
        Y_aux = cat(1,Y_aux,...
            transpose(td(num_seq_train+num_seq_val+i). ... % transpose(td(rand_pos(num_seq_train+num_seq_val+i)). ...
            (predict{(j)})));
    end
    Y_test{(i)} = Y_aux;
end

% Training:

options = trainingOptions(param_net.method, ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',param_net.max_epochs, ...
    'MiniBatchSize',param_net.mini_batch_size, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','none', ... % 'training-progress'
    'Shuffle','every-epoch', ...
    'ValidationData',{X_val,Y_val}, ...
    'ValidationFrequency',10, ...
    'ValidationPatience',30, ...
    'SequenceLength',param_net.sequence_length, ...
    'SequencePaddingDirection','right');

net = trainNetwork(X_train,Y_train,layers,options);

% Subsets:

subsets.train.X = X_train;
subsets.train.Y = Y_train;
subsets.val.X = X_val;
subsets.val.Y = Y_val;
subsets.test.X = X_test;
subsets.test.Y = Y_test;

end