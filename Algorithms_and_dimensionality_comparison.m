%% Paths:

addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\Analysis')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\DataProcessing')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\Plotting')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\td_limblab')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\td_limblab\td_dpca')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\td_limblab\td_gpfa')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\Tools')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\util')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\util\lib\isomap')
addpath('C:\Users\jorge\OneDrive\Escritorio\Proyecto\TrialData_master\util\subfcn')

%% Load data:

load('center_out_data.mat', 'trial_data');

%% Isomap:

td = smoothSignals(trial_data,struct('signals',{'M1_spikes'},'width',0.1));  
%td = removeBadTrials(td);
td = trimTD(td, {'idx_target_on',-5},'idx_trial_end'); 
%td = removeBadNeurons(td,struct('min_fr',0.5));
td_store = td;

clear params
params.algorithm = 'isomap';
params.signals = 'M1_spikes';
params.use_trials = 1:length(td_store);
params.num_dims = 84;
[td_new,info_out] = dimReduce(td_store,params);

save('td_isomap.mat','td_new','info_out');

%% FNN, Kalman and Wiener:

% Initialize:

method = 'linmodel'; % FNN: 'nn', Kalman = 'kalman', Wiener: 'linmodel'
predict = 'vel'; % 'vel', 'pos', 'acc'...
dim_reduce_method = 'fa'; % 'none', 'pca', 'fa', 'ppca', 'isomap'
dims = 71;
percentage_train = 0.9;
polynomial_fit = 3; % for Kalman and Wiener

% Data processing:

td = smoothSignals(trial_data,struct('signals',{'M1_spikes'},'width',0.1));  
%td = removeBadTrials(td);
td = trimTD(td, {'idx_target_on',-5},'idx_trial_end');
%td = removeBadNeurons(td,struct('min_fr',0.5));
td_store = td;

% Dimensionality reduction:

if strcmp(dim_reduce_method,'none') == 1
    dim_reduce_method = 'spikes';
end

if strcmp(dim_reduce_method,'isomap')
    load('td_isomap.mat');
    for i = 1:length(td_store)
        td(i).M1_isomap = td_new(i).M1_isomap(:,1:dims);
    end
elseif strcmp(dim_reduce_method,'spikes') == 0
    clear params
    params.algorithm = dim_reduce_method;
    params.signals = 'M1_spikes';
    params.use_trials = 1:length(td_store);
    params.num_dims = dims;
    [td,info_out] = dimReduce(td_store,params);
end

%%

% Training:

td = getNorm(td,struct('signals',predict));
% td = dupeAndShift(td,strcat('M1_',dim_reduce_method),-(1:5)); 
num_seq_train = fix(percentage_train*length(td));
% rand_pos = randperm(length(td));
td_store = td;
for i = 1:length(td)
    td(i) = td_store(i); % td_store(rand_pos(i));
end
train_idx = 1:num_seq_train;
test_idx = (num_seq_train+1):length(td);

if strcmp(method,'nn')
    model_params = struct('in_signals',{strcat('M1_',dim_reduce_method)},...
        'out_signals',{predict},'model_type',method,'model_name',predict, ...
        'train_idx',train_idx);
else
    model_params = struct('in_signals',{strcat('M1_',dim_reduce_method)},...
        'out_signals',{predict},'model_type',method,'model_name',predict, ...
        'train_idx',train_idx,'polynomial',polynomial_fit);    
end

td = getModel(td,model_params);

x = getSig(td(test_idx),{predict,1});
y = getSig(td(test_idx),{predict,2});
x_pred = getSig(td(test_idx),{strcat(strcat(method,'_'),predict),1});
y_pred = getSig(td(test_idx),{strcat(strcat(method,'_'),predict),2});
vaf_x = compute_vaf(x,x_pred);
vaf_y = compute_vaf(y,y_pred);

figure;
colors = [0 0 1; 1 0 0; 0 0 0; 1 0 0]; % b,r,k,g
ax(1) = subplot(2,1,1,'ColorOrder',colors); hold all;
plot(x,'LineWidth',2);
plot(x_pred,'LineWidth',2);
title(['VAF = ' num2str(vaf_x,3)]);
axis tight;
ax(2) = subplot(2,1,2,'ColorOrder',colors); hold all;
plot(y,'LineWidth',2);
plot(y_pred,'LineWidth',2);
title(['VAF = ' num2str(vaf_y,3)]);
axis tight;
h = legend({'Actual','Predicted'},'Location','SouthOutside');
set(h,'Box','off');
linkaxes(ax,'x');
if strcmp(predict,'pos')
    suptitle('Position (X and Y)');
elseif strcmp(predict,'vel')
    suptitle('Velocity (X and Y)');
elseif strcmp(predict,'acc')
    suptitle('Acceleration (X and Y)');
end

%% LSTM:

% Data processing:

td = smoothSignals(trial_data,struct('signals',{'M1_spikes'},'width',0.1));  
% td = removeBadTrials(td);
td = trimTD(td,{'idx_target_on',-5},'idx_trial_end'); % CHANGE?
% td = removeBadNeurons(td,struct('min_fr',0.5));
td_store = td;

% Training:

clear param;
param.predict = {'vel'};
param.dim_reduce_method = 'isomap';
param.dims = 5;
param.percentage_train = 0.8;
param.percentage_val = 0.1;
param.mini_batch_size = 16;
param.max_epochs = 60;
param.num_hidden_layers = 1;
param.num_hidden_units = [40];
param.dropout = [0.2];
param.method = 'sgdm';
param.sequence_length = 'shortest';

if strcmp(param.dim_reduce_method,'isomap')
    load('td_isomap.mat');
    for i = 1:length(td_store)
        td_store(i).M1_isomap = td_new(i).M1_isomap(:,1:param.dims);
    end
end

[net,subsets] = runLSTM(td_store,param);

X_train = subsets.train.X;
Y_train = subsets.train.Y;
X_test = subsets.test.X;
Y_test = subsets.test.Y;

% Predictions:

Y_pred_train = cell(length(Y_train),1);
for i = 1:length(Y_train)
    [~,Y_pred_train{(i)}] = predictAndUpdateState(net,X_train{(i)},...
        'ExecutionEnvironment','cpu');
end
Y_pred_test = cell(length(Y_test),1);
for i = 1:length(Y_test)
    [~,Y_pred_test{(i)}] = predictAndUpdateState(net,X_test{(i)},...
        'ExecutionEnvironment','cpu');
end

%%

% VAF and plots:

clear vaf
for i = 1:length(param.predict)
    aux = [];
    aux_pred = [];
    for j = 1:length(Y_train)
        aux = cat(2,aux,Y_train{(j)}((2*i-1):(2*i),:));
        aux_pred = cat(2,aux_pred,Y_pred_train{(j)}((2*i-1):(2*i),:));
    end
    vaf.train.(param.predict{(i)}).X = compute_vaf(transpose(aux(1,:)),...
        transpose(aux_pred(1,:)));
    vaf.train.(param.predict{(i)}).Y = compute_vaf(transpose(aux(2,:)),...
        transpose(aux_pred(2,:)));
    aux = [];
    aux_pred = [];
    for j = 1:length(Y_test)
        aux = cat(2,aux,Y_test{(j)}((2*i-1):(2*i),:));
        aux_pred = cat(2,aux_pred,Y_pred_test{(j)}((2*i-1):(2*i),:));
    end
    vaf.test.(param.predict{(i)}).X = compute_vaf(transpose(aux(1,:)),...
        transpose(aux_pred(1,:)));
    vaf.test.(param.predict{(i)}).Y = compute_vaf(transpose(aux(2,:)),...
        transpose(aux_pred(2,:)));
    
    fprintf('\n');
    disp(strcat('vaf_train_',param.predict{(i)}));
    disp(vaf.train.(param.predict{(i)}));
    disp(strcat('vaf_test_',param.predict{(i)}));
    disp(vaf.test.(param.predict{(i)}));
    
    figure(i);
    colors = [0 0 1; 1 0 0; 0 0 0; 1 0 0]; % b,r,k,g
    ax(1) = subplot(2,1,1,'ColorOrder',colors); hold all;
    plot(aux(1,:),'LineWidth',2);
    plot(aux_pred(1,:),'LineWidth',2);
    title(['VAF = ' num2str(vaf.test.(param.predict{(i)}).X,3)]);
    axis tight;
    ax(2) = subplot(2,1,2,'ColorOrder',colors); hold all;
    plot(aux(2,:),'LineWidth',2);
    plot(aux_pred(2,:),'LineWidth',2);
    title(['VAF = ' num2str(vaf.test.(param.predict{(i)}).Y,3)]);
    axis tight;
    h = legend({'Actual','Predicted'},'Location','SouthOutside');
    set(h,'Box','off');
    linkaxes(ax,'x');
    if strcmp(param.predict{(i)},'pos')
        suptitle('Position (X and Y)');
    elseif strcmp(param.predict{(i)},'vel')
        suptitle('Velocity (X and Y)');
    elseif strcmp(param.predict{(i)},'acc')
        suptitle('Acceleration (X and Y)');
    end
end

%% Explained variances:

dims = [1 2 3 4 5 7 9 13 30 50 70];

exp_var_pca = [0.2103 0.3169 0.3845 0.4465 0.5015 0.5659 0.6184 0.6814 0.8292 0.9224 0.9816];
exp_var_isomap = [0.5668 0.7403 0.8530 0.9063 0.9112 0.9177 0.9250 0.9303 0.9314 0.9275 0.9238]; % 1 - residual_variance
figure;
plot(dims,exp_var_pca,'r','LineWidth',1);
hold on;
plot(dims,exp_var_isomap,'b','LineWidth',1);
title('Explained variance');
axis tight;
h = legend({'PCA','Isomap'},'Location','SouthOutside');
set(h,'Box','off');
ylim([0 1]);

%% Plots velocity:

vaf_vel_lstm = [0.4204 0.6487 0.9044 0.9240 0.9177 0.9253 0.9285 0.9256 0.9121 0.9128 0.9212;
    0.3944 0.6654 0.8405 0.9301 0.9142 0.9261 0.9250 0.9309 0.9299 0.9204 0.9288;
    0.1977 0.6472 0.7852 0.9215 0.9189 0.9124 0.9204 0.9289 0.9172 0.9248 0.9207]; % 1: isomap, 2: pca, 3: fa
std_vaf_vel_lstm = [0.0684 0.0882 0.0163 0.0085 0.0045 0.0031 0.0119 0.0064 0.0107 0.0057 0.0041;
    0.0104 0.1936 0.0401 0.0038 0.0087 0.0043 0.0031 0.0078 0.0068 0.0079 0.0070;
    0.1542 0.0394 0.0758 0.0181 0.0099 0.0134 0.0043 0.0068 0.0127 0.0056 0.0037]; % 1: isomap, 2: pca, 3: fa / n = 3

vaf_vel_fnn = [0.0167 0.4349 0.8548 0.8773 0.8820 0.8872 0.8889 0.8939 0.8800 0.8811 0.8497;
    0.1970 0.5367 0.6468 0.7864 0.8020 0.8106 0.8269 0.8608 0.8640 0.8385 0.8503;
    0.1705 0.4905 0.6383 0.8012 0.7876 0.8277 0.8289 0.8633 0.8700 0.8534 0.8321]; % 1: isomap, 2: pca, 3: fa
std_vaf_vel_fnn = [0.0110 0.0306 0.0275 0.0101 0.0217 0.0019 0.0169 0.0034 0.0181 0.0172 0.0163;
    0.0350 0.0349 0.0141 0.0155 0.0166 0.0113 0.0038 0.0251 0.0134 0.0496 0.0103;
    0.0050 0.0425 0.0152 0.0088 0.0085 0.0130 0.0102 0.0040 0.0026 0.0219 0.0221]; % 1: isomap, 2: pca, 3: fa / n = 3

vaf_vel_kalman = [-0.0046 0.2282 0.4642 0.4705 0.4775 0.4942 0.4879 0.5094 0.5851 0.5964 0.6049;
   -0.1541 0.0771 0.2257 0.4117 0.5229 0.5092 0.5604 0.6379 0.6494 0.6508 0.6651;
   -0.0538 0.0898 0.2325 0.4661 0.4774 0.5101 0.5307 0.6406 0.6540 0.6518 0.6608]; % 1: isomap, 2: pca, 3: fa
std_vaf_vel_kalman = [0.0055 0.0123 0.0017 0.0208 0.0070 0.0157 0.0107 0.0084 0.0085 0.0188 0.0100;
    0.0253 0.0268 0.0311 0.0800 0.0184 0.0019 0.0261 0.0214 0.0223 0.0126 0.0157;
    0.0335 0.0376 0.0129 0.0333 0.0189 0.0260 0.0356 0.0016 0.0077 0.0260 0.0411]; % 1: isomap, 2: pca, 3: fa / n = 3

vaf_vel_wiener = [0.0037 0.2907 0.5935 0.5856 0.5939 0.5945 0.6047 0.6090 0.6209 0.6460 0.6621;
    0.0203 0.2116 0.3079 0.4733 0.5048 0.5479 0.5915 0.6430 0.6825 0.7028 0.6997;
    0.0234 0.1966 0.2997 0.4894 0.5186 0.5428 0.5614 0.6664 0.6912 0.7024 0.7112]; % 1: isomap, 2: pca, 3: fa
std_vaf_vel_wiener = [0.0083 0.0205 0.0063 0.0111 0.0111 0.0063 0.0058 0.0077 0.0122 0.0069 0.0048;
    0.0212 0.0314 0.0118 0.0217 0.0154 0.0150 0.0179 0.0170 0.0083 0.0108 0.0130;
    0.0112 0.0040 0.0185 0.0327 0.0098 0.0054 0.0277 0.0091 0.0070 0.0079 0.0073]; % 1: isomap, 2: pca, 3: fa / n = 3

%%

figure;
ax(1) = subplot(1,4,1); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_vel_lstm(1,:),std_vaf_vel_lstm(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_lstm(2,:),std_vaf_vel_lstm(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_lstm(3,:),std_vaf_vel_lstm(3,:), ...
    'lineprops',{'k'})
title('LSTM');
axis tight;
ylim([0 1]);
ylabel('VAF');
xlabel('Dimensions');
ax(2) = subplot(1,4,2); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_vel_fnn(1,:),std_vaf_vel_fnn(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(2,:),std_vaf_vel_fnn(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(3,:),std_vaf_vel_fnn(3,:), ...
    'lineprops',{'k'})
title('FNN');
axis tight;
ylim([0 1]);
xlabel('Dimensions');
ax(3) = subplot(1,4,3); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_vel_kalman(1,:),std_vaf_vel_kalman(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(2,:),std_vaf_vel_kalman(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(3,:),std_vaf_vel_kalman(3,:), ...
    'lineprops',{'k'})
title('Kalman');
axis tight;
ylim([0 1]);
xlabel('Dimensions');
ax(4) = subplot(1,4,4); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_vel_wiener(1,:),std_vaf_vel_wiener(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(2,:),std_vaf_vel_wiener(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(3,:),std_vaf_vel_wiener(3,:), ...
    'lineprops',{'k'})
title('Wiener');
axis tight;
ylim([0 1]);
xlabel('Dimensions');
linkaxes(ax,'y');
suptitle('VELOCITY PREDICTION PERFORMANCE');
h = legend({'Isomap','PCA','FA'}');
newPosition = [0.779 0.12 0.1 0.1];
newUnits = 'normalized';
set(h,'Position', newPosition,'Units', newUnits);

%%

figure(1);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_vel_lstm(1,:),std_vaf_vel_lstm(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_lstm(2,:),std_vaf_vel_lstm(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_lstm(3,:),std_vaf_vel_lstm(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (LSTM - velocity)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(2);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_vel_fnn(1,:),std_vaf_vel_fnn(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(2,:),std_vaf_vel_fnn(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(3,:),std_vaf_vel_fnn(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (FNN - velocity)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(3);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_vel_kalman(1,:),std_vaf_vel_kalman(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(2,:),std_vaf_vel_kalman(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(3,:),std_vaf_vel_kalman(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (Kalman - velocity)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(4);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_vel_wiener(1,:),std_vaf_vel_wiener(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(2,:),std_vaf_vel_wiener(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(3,:),std_vaf_vel_wiener(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (Wiener - velocity)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(5);
shadedErrorBar(dims,vaf_vel_lstm(1,:),std_vaf_vel_lstm(1,:),...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(1,:),std_vaf_vel_fnn(1,:),...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(1,:),std_vaf_vel_kalman(1,:),...
    'lineprops',{'k'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(1,:),std_vaf_vel_wiener(1,:),...
    'lineprops',{'g'})
title('Isomap: variance-accounted-for vs dimensions (velocity)');
axis tight;
h = legend({'LSTM','FNN','Kalman','Wiener'},'Location','SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(6);
shadedErrorBar(dims,vaf_vel_lstm(2,:),std_vaf_vel_lstm(2,:),...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(2,:),std_vaf_vel_fnn(2,:),...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(2,:),std_vaf_vel_kalman(2,:),...
    'lineprops',{'k'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(2,:),std_vaf_vel_wiener(2,:),...
    'lineprops',{'g'})
title('PCA: variance-accounted-for vs dimensions (velocity)');
axis tight;
h = legend({'LSTM','FNN','Kalman','Wiener'},'Location','SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(7);
shadedErrorBar(dims,vaf_vel_lstm(3,:),std_vaf_vel_lstm(3,:),...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_vel_fnn(3,:),std_vaf_vel_fnn(3,:),...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_vel_kalman(3,:),std_vaf_vel_kalman(3,:),...
    'lineprops',{'k'})
hold on;
shadedErrorBar(dims,vaf_vel_wiener(3,:),std_vaf_vel_wiener(3,:),...
    'lineprops',{'g'})
title('FA: variance-accounted-for vs dimensions (velocity)');
axis tight;
h = legend({'LSTM','FNN','Kalman','Wiener'},'Location','SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

%% Plots position:

dims = [1 2 3 4 5 7 9 13 30 50 70];

exp_var_pca = [0.2103 0.3169 0.3845 0.4465 0.5015 0.5659 0.6184 0.6814 0.8292 0.9224 0.9816];

vaf_pos_lstm = [0.2962 0.6967 0.9709 0.9742 0.9762 0.9768 0.9755 0.9722 0.9688 0.9758 0.9706;
    0.3128 0.8589 0.9062 0.9573 0.9656 0.9740 0.9656 0.9727 0.9689 0.9661 0.9675;
    0.1112 0.7042 0.8943 0.9436 0.9769 0.9660 0.9673 0.9670 0.9720 0.9779 0.9713]; % 1: isomap, 2: pca, 3: fa
std_vaf_pos_lstm = [0.0414 0.0573 0.0041 0.0053 0.0049 0.0063 0.0018 0.0035 0.0020 0.0032 0.0036;
    0.0767 0.0354 0.0330 0.0150 0.0057 0.0020 0.0069 0.0084 0.0032 0.0039 0.0022;
    0.1262 0.1103 0.0116 0.0286 0.0039 0.0048 0.0084 0.0021 0.0049 0.0019 0.0015]; % 1: isomap, 2: pca, 3: fa / n = 3

vaf_pos_fnn = [0.2928 0.4868 0.8909 0.8966 0.9030 0.9210 0.9237 0.9295 0.9328 0.9221 0.9201;
    0.0411 0.3143 0.3729 0.5938 0.6514 0.7373 0.7875 0.8874 0.8970 0.8814 0.8748;
    0.0331 0.2824 0.3664 0.6023 0.6746 0.7255 0.8349 0.8805 0.8965 0.8857 0.8609]; % 1: isomap, 2: pca, 3: fa
std_vaf_pos_fnn = [0.0168 0.0176 0.0228 0.0061 0.0067 0.0082 0.0063 0.0038 0.0051 0.0088 0.0054;
    0.0051 0.0269 0.0378 0.0265 0.0204 0.0314 0.0010 0.0023 0.0052 0.0080 0.0070;
    0.0119 0.0468 0.0292 0.0295 0.0034 0.0249 0.0230 0.0165 0.0103 0.0054 0.0050]; % 1: isomap, 2: pca, 3: fa / n = 3

vaf_pos_kalman = [-0.2233 -0.1154 0.2277 0.3212 0.3551 0.3730 0.3831 0.3831 0.3892 0.3960 0.3781;
   -0.1347 0.0890 0.1683 0.1980 0.1465 0.1996 0.1994 0.3400 0.3684 0.3892 0.3884;
   -0.0584 0.1619 0.2045 0.2153 0.2188 0.2015 0.2292 0.3630 0.3933 0.3800 0.3813]; % 1: isomap, 2: pca, 3: fa
std_vaf_pos_kalman = [0.0614 0.2106 0.0197 0.0170 0.0118 0.0351 0.0036 0.0052 0.0181 0.0093 0.0151;
    0.0549 0.0729 0.0218 0.0192 0.0157 0.0205 0.0022 0.0126 0.0273 0.0042 0.0147;
    0.0455 0.0120 0.0391 0.0260 0.0263 0.0504 0.0520 0.0059 0.0182 0.0283 0.0118]; % 1: isomap, 2: pca, 3: fa / n = 3

vaf_pos_wiener = [0.1072 0.1865 0.4253 0.5102 0.5461 0.6711 0.6850 0.7095 0.7218 0.7067 0.7288;
    0.0078 0.1384 0.1338 0.2341 0.2924 0.3827 0.4317 0.6541 0.7176 0.7390 0.7463;
    0.0002 0.1407 0.1260 0.2595 0.3097 0.3406 0.4567 0.6525 0.7335 0.7429 0.7411]; % 1: isomap, 2: pca, 3: fa
std_vaf_pos_wiener = [0.0331 0.0487 0.0127 0.0127 0.0125 0.0083 0.0076 0.0203 0.0278 0.0163 0.0174;
    0.0231 0.0220 0.0213 0.0069 0.0321 0.0076 0.0132 0.0149 0.0082 0.0057 0.0045;
    0.0244 0.0087 0.0116 0.0133 0.0177 0.0175 0.0142 0.0054 0.0064 0.0104 0.0118]; % 1: isomap, 2: pca, 3: fa / n = 3

%%

figure;
ax(1) = subplot(1,4,1); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_pos_lstm(1,:),std_vaf_pos_lstm(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_lstm(2,:),std_vaf_pos_lstm(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_lstm(3,:),std_vaf_pos_lstm(3,:), ...
    'lineprops',{'k'})
title('LSTM');
axis tight;
ylim([-0.4 1]);
ylabel('VAF');
xlabel('Dimensions');
ax(2) = subplot(1,4,2); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_pos_fnn(1,:),std_vaf_pos_fnn(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(2,:),std_vaf_pos_fnn(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(3,:),std_vaf_pos_fnn(3,:), ...
    'lineprops',{'k'})
title('FNN');
axis tight;
ylim([-0.4 1]);
xlabel('Dimensions');
ax(3) = subplot(1,4,3); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_pos_kalman(1,:),std_vaf_pos_kalman(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(2,:),std_vaf_pos_kalman(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(3,:),std_vaf_pos_kalman(3,:), ...
    'lineprops',{'k'})
title('Kalman');
axis tight;
ylim([-0.4 1]);
xlabel('Dimensions');
ax(4) = subplot(1,4,4); hold all;
%plot(dims,exp_var_pca,'LineWidth',1);
%hold on;
shadedErrorBar(dims,vaf_pos_wiener(1,:),std_vaf_pos_wiener(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(2,:),std_vaf_pos_wiener(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(3,:),std_vaf_pos_wiener(3,:), ...
    'lineprops',{'k'})
title('Wiener');
axis tight;
ylim([-0.4 1]);
xlabel('dimensions');
linkaxes(ax,'y');
suptitle('Position Prediction Performance');
h = legend({'Isomap','PCA','FA'}');
newPosition = [0.779 0.12 0.1 0.1];
newUnits = 'normalized';
set(h,'Position', newPosition,'Units', newUnits);

%%

figure(8);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_pos_lstm(1,:),std_vaf_pos_lstm(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_lstm(2,:),std_vaf_pos_lstm(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_lstm(3,:),std_vaf_pos_lstm(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (LSTM - position)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(9);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_pos_fnn(1,:),std_vaf_pos_fnn(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(2,:),std_vaf_pos_fnn(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(3,:),std_vaf_pos_fnn(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (FNN - position)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(10);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_pos_kalman(1,:),std_vaf_pos_kalman(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(2,:),std_vaf_pos_kalman(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(3,:),std_vaf_pos_kalman(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (Kalman - position)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(11);
plot(dims,exp_var_pca,'LineWidth',1);
hold on;
shadedErrorBar(dims,vaf_pos_wiener(1,:),std_vaf_pos_wiener(1,:), ...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(2,:),std_vaf_pos_wiener(2,:), ...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(3,:),std_vaf_pos_wiener(3,:), ...
    'lineprops',{'k'})
title('Variance vs dimensions (Wiener - position)');
axis tight;
h = legend({'Explained (PCA)','Accounted-for (Isomap)', ...
    'Accounted-for (PCA)','Accounted-for (FA)'},'Location', ...
    'SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(12);
shadedErrorBar(dims,vaf_pos_lstm(1,:),std_vaf_pos_lstm(1,:),...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(1,:),std_vaf_pos_fnn(1,:),...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(1,:),std_vaf_pos_kalman(1,:),...
    'lineprops',{'k'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(1,:),std_vaf_pos_wiener(1,:),...
    'lineprops',{'g'})
title('Isomap: variance-accounted-for vs dimensions (position)');
axis tight;
h = legend({'LSTM','FNN','Kalman','Wiener'},'Location','SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(13);
shadedErrorBar(dims,vaf_pos_lstm(2,:),std_vaf_pos_lstm(2,:),...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(2,:),std_vaf_pos_fnn(2,:),...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(2,:),std_vaf_pos_kalman(2,:),...
    'lineprops',{'k'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(2,:),std_vaf_pos_wiener(2,:),...
    'lineprops',{'g'})
title('PCA: variance-accounted-for vs dimensions (position)');
axis tight;
h = legend({'LSTM','FNN','Kalman','Wiener'},'Location','SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);

figure(14);
shadedErrorBar(dims,vaf_pos_lstm(3,:),std_vaf_pos_lstm(3,:),...
    'lineprops',{'b'})
hold on;
shadedErrorBar(dims,vaf_pos_fnn(3,:),std_vaf_pos_fnn(3,:),...
    'lineprops',{'r'})
hold on;
shadedErrorBar(dims,vaf_pos_kalman(3,:),std_vaf_pos_kalman(3,:),...
    'lineprops',{'k'})
hold on;
shadedErrorBar(dims,vaf_pos_wiener(3,:),std_vaf_pos_wiener(3,:),...
    'lineprops',{'g'})
title('FA: variance-accounted-for vs dimensions (position)');
axis tight;
h = legend({'LSTM','FNN','Kalman','Wiener'},'Location','SouthOutside');
set(h,'Box','off');
ylim([-0.4 1]);