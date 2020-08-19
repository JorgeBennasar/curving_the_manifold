%% Paths:

clear;
clc;

%%

addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\Analysis')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\DataProcessing')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\Plotting')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\td_limblab')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\td_limblab\td_dpca')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\td_limblab\td_gpfa')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\Tools')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\util')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\util\subfcn')
addpath('C:\Users\jorge\OneDrive\Escritorio\linear_vs_nonlinear_manifolds\myLSTM')

%% Load and process data:

monkey = 'Chewie_CO_CS_2016-10-21'; 
% 'Mihili_CO_VR_2014-03-06' (M_1) / 'Mihili_CO_FF_2014-02-17' (M_2) / 
    % 'Mihili_CO_VR_2014-03-03' (M_3) / 'Mihili_CO_VR_2014-03-04' (M_4) /
    % 'Chewie_CO_CS_2016-10-21' (C_1) / 'Chewie_CO_FF_2016-10-07' (C_2) /
    % 'Chewie_CO_VR_2016-10-06' (C_3) / 'Chewie_CO_VR_2016-09-29' (C_4) /
    % 'Chewie_CO_CS_2015-03-19' (C2_1) / 'Chewie_CO_CS_2015-03-11' (C2_2) /
    % 'Chewie_CO_CS_2015-03-12' (C2_3) / 'Chewie_CO_CS_2015-03-13' (C2_4) 
selection = 'M1'; % 'M1' / 'PMd' / 'M1_and_PMd'
min_firing_rate = 0.1;

if strcmp(monkey,'Mihili_CO_VR_2014-03-03')
    load('Mihili_CO_VR_2014-03-03.mat', 'trial_data');
elseif strcmp(monkey,'Mihili_CO_VR_2014-03-06')
    load('Mihili_CO_VR_2014-03-06.mat', 'trial_data');
elseif strcmp(monkey,'Mihili_CO_FF_2014-02-17')
    load('Mihili_CO_FF_2014-02-17.mat', 'trial_data');
elseif strcmp(monkey,'Mihili_CO_VR_2014-03-04')
    load('Mihili_CO_VR_2014-03-04.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_FF_2016-10-07')    
    load('Chewie_CO_FF_2016-10-07.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_CS_2016-10-21')
    load('Chewie_CO_CS_2016-10-21.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_VR_2016-10-06')
    load('Chewie_CO_VR_2016-10-06.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_VR_2016-09-29')
    load('Chewie_CO_VR_2016-09-29.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_CS_2015-03-19')
    load('Chewie_CO_CS_2015-03-19.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_CS_2015-03-11')
    load('Chewie_CO_CS_2015-03-11.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_CS_2015-03-12')
    load('Chewie_CO_CS_2015-03-12.mat', 'trial_data');
elseif strcmp(monkey,'Chewie_CO_CS_2015-03-13')
    load('Chewie_CO_CS_2015-03-13.mat', 'trial_data');
end

num_bins = 3; % 30 ms
width_smooth = 0.1;
preparation_bins = [-7, -1]; % -210 ms to -30 ms from idx_movement_on
simulation_bins = [-12, 14]; 
td = removeBadTrials(trial_data);
td = binTD(td,num_bins);
td = removeBadNeurons(td,struct('min_fr',min_firing_rate));

td = sqrtTransform(td,struct('signals',{'M1_spikes'}));
%td = sqrtTransform(td,struct('signals',{'PMd_spikes'}));
td = smoothSignals(td,struct('signals',{'M1_spikes'},'width', ...
    width_smooth));  
%td = smoothSignals(td,struct('signals',{'PMd_spikes'},'width', ...
%    width_smooth));

% Trial removal:

l = length(td);
t_1 = [];
t_2 = [];
t_3 = [];
t_4 = [];
t_5 = [];
counter = 0;
for i = 1:l
    t_1_now = td(i-counter).idx_target_on;
    t_2_now = td(i-counter).idx_movement_on;
    if (t_1_now < 3000) == 0 || (t_2_now < 3000) == 0
        td(i-counter) = [];
        counter = counter + 1;
    else
        t_1 = [t_1 td(i-counter).idx_target_on];
        t_2 = [t_2 td(i-counter).idx_movement_on];
        t_3 = [t_3 td(i-counter).idx_movement_on- ...
            td(i-counter).idx_target_on];
        t_4 = [t_4 td(i-counter).idx_go_cue];
        t_5 = [t_5 td(i-counter).idx_trial_end];
    end
end

counter = 0;
for i = 1:length(td)
    threshold_x = 0.125*max(abs(td(i-counter).vel(:,1)));
    threshold_y = 0.125*max(abs(td(i-counter).vel(:,2)));
    a = abs(td(i-counter).vel(t_2(i-counter)+1,1) - ...
        td(i-counter).vel(t_2(i-counter),1));
    b = abs(td(i-counter).vel(t_2(i-counter)+1,2) - ...
        td(i-counter).vel(t_2(i-counter),2));
    if (a < threshold_x && b < threshold_y) || (t_3(i-counter) < ...
            -simulation_bins(1)-1) || (t_2(i-counter) <= ...
            -simulation_bins(1)) || size(td(i-counter).vel,1) - ...
            t_2(i-counter) < simulation_bins(2)+1
        td(i-counter) = [];
        t_1(i-counter) = [];
        t_2(i-counter) = [];
        t_3(i-counter) = [];
        t_4(i-counter) = [];
        t_5(i-counter) = [];
        counter = counter + 1;
    elseif strcmp(td(i-counter).epoch,'BL') == 0
        td(i-counter) = [];
        t_1(i-counter) = [];
        t_2(i-counter) = [];
        t_3(i-counter) = [];
        t_4(i-counter) = [];
        t_5(i-counter) = [];
        counter = counter + 1;
    elseif strcmp(td(i-counter).task,'CO') == 0
        td(i-counter) = [];
        t_1(i-counter) = [];
        t_2(i-counter) = [];
        t_3(i-counter) = [];
        t_4(i-counter) = [];
        t_5(i-counter) = [];
        counter = counter + 1;
    end
end

disp(['Removed ' num2str(counter) ' trials']);

td_p = trimTD(td,{'idx_movement_on',preparation_bins(1)}, ...
    {'idx_movement_on',preparation_bins(2)}); 
td_m = trimTD(td,{'idx_movement_on',simulation_bins(1)}, ...
    {'idx_movement_on',simulation_bins(2)}); 

% Target detection:

target_index = ones(1,8);
counter = 1;
target_counter = zeros(1,8);
targets = zeros(1,length(td_m));
for i = 1:length(td_m)
    if ismember(td_m(i).target_direction,target_index) == 0
        target_index(counter) = td_m(i).target_direction;
        counter = counter + 1;
    end
    idx = find(target_index == td_m(i).target_direction);
    target_counter(idx) = target_counter(idx) + 1;
    targets(i) = idx;
end

% Data selection:

if strcmp(selection,'M1')
    neurons = size(td_m(1).M1_spikes,2);
elseif strcmp(selection,'PMd')
    neurons = size(td_m(1).PMd_spikes,2);
elseif strcmp(selection,'M1_and_PMd')
    a = size(td_m(1).M1_spikes,2);
    b = size(td_m(1).PMd_spikes,2);
    neurons = 2*min(a,b);
end

time_m = simulation_bins(2)-simulation_bins(1)+1;
time_p = preparation_bins(2)-preparation_bins(1)+1;
N_m = zeros(neurons,length(td_m),time_m);
N_p = zeros(neurons,length(td_p),time_p);
Y = zeros(2,length(td_p),time_m);

for i = 1:length(td_m)
    for j = 1:time_m
        if strcmp(selection,'M1')
            N_m(:,i,j) = td_m(i).M1_spikes(j,1:neurons);
        elseif strcmp(selection,'PMd')
            N_m(:,i,j) = td_m(i).PMd_spikes(j,1:neurons);
        elseif strcmp(selection,'M1_and_PMd')
            N_m(1:neurons/2,i,j) = td_m(i).M1_spikes(j,1:neurons/2);
            N_m(neurons/2+1:end,i,j) = td_m(i).PMd_spikes(j,1:neurons/2);
        end
        Y(:,i,j) = td_m(i).vel(j,:);
    end
    for j = 1:time_p
        if strcmp(selection,'M1')
            N_p(:,i,j) = td_p(i).M1_spikes(j,1:neurons);
        elseif strcmp(selection,'PMd')
            N_p(:,i,j) = td_p(i).PMd_spikes(j,1:neurons);
        elseif strcmp(selection,'M1_and_PMd')
            N_p(1:neurons/2,i,j) = td_p(i).M1_spikes(j,1:neurons/2);
            N_p(neurons/2+1:end,i,j) = td_p(i).PMd_spikes(j,1:neurons/2);
        end
    end 
end

%% Inputs:

t = 25;

dims = 3;
[N_p_pca,~] = get_pca(N_p,dims);
X = zeros(dims+1,size(N_m,2),t);

for i = 1:size(N_m,2)
    input_p = zeros(dims,t); % preparatory input
    input_h = zeros(1,t); % hold input
    for j = 1:t
        if j < t-14
            input_h(j) = 0.5;
            if j >= t-21
                for k = 1:dims
                    input_p(k,j) = sum(squeeze(N_p_pca(k,i,:)));
                end
            end
        end
    end
    X(:,i,:) = [input_p; input_h]; % total input
end

Y = Y(:,:,end-t+1:end);
    
%% Target selection: 

target_sel = 1:8;
index_target_sel = [];
new_targets = [];
for j = target_sel
    index_target_sel = [index_target_sel find(targets == j)];
    new_targets = [new_targets j*ones(1,length(find(targets == j)))];
end 

data.X = X(:,index_target_sel,:);
data.Y = Y(:,index_target_sel,:);
data.targets = new_targets;
data.target_sel = target_sel;

%% Train:

percentage_train = 0.7; % w.r.t. the target with less trials

clear param;
param.m_train_per_target = fix(min(target_counter(target_sel))*percentage_train);
param.mini_batch_size = 8;
param.num_epochs = 10000;
param.stop_condition = 8; 
param.n_hidden = 100;
param.beta_1 = 0.9;
param.beta_2 = 0.999;
param.epsilon = 1e-8;
param.learning_rate = 0.005;
param.learning_rate_change = 'no';
param.learning_rate_rule = 1/3; % learning_rate_now = learning_rate/i^(learning_rate_rule)
param.optimization = 'adam';
param.transfer_learning = 'false';
param.transfer_param = 0;
param.r_or_c = 'regression';
param.lambda = 0; % lambda = 0 for no L2 regularization
param.connectivity = 1;
param.amp_noise = 0;
param.noise_samples = zeros(1,t);
param.mode = 1; % 1 for new trials, 2 for the previous ones
param.correlation_reg = 0; 

if param.mode == 1
    [net,cost_train,subsets] = run_my_LSTM(data,param);
elseif param.mode == 2
    data.x_train = subsets.train.X;
    data.y_train = subsets.train.Y;
    [net,cost_train,~] = run_my_LSTM(data,param); 
else
    disp('###########');
    disp('Incorrect mode, choose 1 or 2');
    disp('###########');
    return;
end

disp('###########');
disp(['Cost train:' ' ' num2str(cost_train)]);
X_test = subsets.test.X;
Y_test = subsets.test.Y;
index_test = subsets.test.index;
disp('###########');

% Data processing:

A_test = zeros(param.n_hidden,size(Y_test,2),t);
N_test = zeros(size(N_m,1),size(Y_test,2),t);
Y_pred = zeros(size(Y_test));

for i = 1:size(Y_test,2)
    [Y_pred_t, A_t] = LSTM_predict(X_test(:,i,:),net,param.r_or_c, ...
        param.amp_noise, param.noise_samples);
    Y_pred(:,i,:) = Y_pred_t;
    A_test(:,i,:) = A_t;
    index = index_target_sel(index_test(i));
    N_test(:,i,:) = N_m(:,index,end-t+1:end);
end

% Conversion of activations to all positive values:

A_test = abs(A_test);

% Processing like neural activity:

A_test = sqrt(A_test);

for i = 1:size(Y_test,2)
    A_t = smooth_data(transpose(squeeze(A_test(:,i,:))), ...
        td(1).bin_size,width_smooth);
    for j = 1:size(A_test,1)
        for k = 1:size(A_test,3)
            A_test(j,i,k) = A_t(k,j);
        end
    end
end

%{ 

%% Elimination of outliers in neurons:

mean_max = mean(max(max(N_test(:,:,:),[],2),[],3));
std_max = sqrt(var(max(max(N_test(:,:,:),[],2),[],3)));

counter = 0;
for i = 1:size(N_test,1)
    max_neuron = max(max(N_test(i-counter,:,:),[],2),[],3);
    if max_neuron > mean_max+std_max
        N_test(i-counter,:,:) = [];
        counter = counter + 1;
    end
end

disp(['Removed ' num2str(counter) ' neurons']);

%}

%% Velocity:

time_plot = linspace(1,size(N_test,3)*size(Y_test,2),size(N_test,3)*size(Y_test,2));

Y_test_plot = zeros(2,size(N_test,3)*size(Y_test,2));
Y_pred_plot = zeros(2,size(N_test,3)*size(Y_test,2));

for i = 1:size(N_test,3)
    for j = 1:size(Y_test,2)
        Y_test_plot(1,(j-1)*size(N_test,3)+i) = Y_test(1,j,i);
        Y_pred_plot(1,(j-1)*size(N_test,3)+i) = Y_pred(1,j,i);
        Y_test_plot(2,(j-1)*size(N_test,3)+i) = Y_test(2,j,i);
        Y_pred_plot(2,(j-1)*size(N_test,3)+i) = Y_pred(2,j,i);
    end
end

vaf_x = compute_vaf(transpose(Y_test_plot(1,:)), ...
    transpose(Y_pred_plot(1,:)));
vaf_y = compute_vaf(transpose(Y_test_plot(2,:)), ...
    transpose(Y_pred_plot(2,:)));
mean_vaf = mean([vaf_x vaf_y]);

figure;
ax(1) = subplot(2,1,1); hold all;
plot(Y_test_plot(1,:),'Color',[1 0 0],'LineWidth',1.5);
plot(Y_pred_plot(1,:),'Color',[0 0 1],'LineWidth',1.5);
title(['X (VAF = ' num2str(vaf_x,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
ax(2) = subplot(2,1,2); hold all;
plot(Y_test_plot(2,:),'Color',[1 0 0],'LineWidth',1.5);
plot(Y_pred_plot(2,:),'Color',[0 0 1],'LineWidth',1.5);
title(['Y (VAF = ' num2str(vaf_y,3) ')']);
xlabel('time');
ylabel('velocity');
axis tight;
h = legend({'Actual','Predicted'});
newPosition = [0.779 0.12 0.1 0.1];
newUnits = 'normalized';
set(h,'Position', newPosition,'Units', newUnits);
linkaxes(ax,'x');
suptitle('LSTM Velocity Prediction (Test)');

%% Normalization of times:

N_norm = N_test;
A_norm = A_test;

N_norm = N_norm/max(N_norm,[],'all');
A_norm = A_norm/max(A_norm,[],'all');

%% PSTHs:

PSTH = obtain_PSTH(A_norm,new_targets,index_test);
PSTH_N = obtain_PSTH(N_norm,new_targets,index_test);

min_length = min(size(N_norm,1),size(A_norm,1));
targ = 6;

idx = randperm(size(PSTH,1));
idx_N = randperm(size(PSTH_N,1));

PSTH = squeeze(PSTH(idx(1:min_length),targ,:));
PSTH_N = squeeze(PSTH_N(idx_N(1:min_length),targ,:));

PSTH_mean = mean(PSTH,2);
PSTH_N_mean = mean(PSTH_N,2);

[~,idx] = sort(PSTH_mean);
PSTH = PSTH(idx,:);
[~,idx] = sort(PSTH_N_mean);
PSTH_N = PSTH_N(idx,:);

figure;

subplot(2,1,1);
for i = 1:min_length
    plot(PSTH(i,:),'Color',[0 0.5 1],'LineWidth',0.5);
    hold on;
end
xline(t-14,'k--','LineWidth',2);
ylim([-0.5 1.5]);
xlabel('time');
ylabel('rate of firing');
title('RNN units');

subplot(2,1,2);
for i = 1:min_length
    plot(PSTH_N(i,:),'Color',[0 1 0.5],'LineWidth',0.5);
    hold on;
end
xline(t-14,'k--','LineWidth',2);
ylim([-0.5 1.5]);
title('Neurons');
xlabel('time');
ylabel('rate of firing');
suptitle('Comparison between normalized PSTHs for one target');

figure;
subplot(2,1,1);
image(PSTH,'CDataMapping','scaled');
hold on;
xline(t-14,'k','LineWidth',3);
colorbar;
title('Neural network');
xlabel('time');
ylabel('units');
subplot(2,1,2);
image(PSTH_N,'CDataMapping','scaled');
hold on;
xline(t-14,'k','LineWidth',3);
colorbar;
title('Motor cortex');
xlabel('time');
ylabel('neurons');
suptitle('Comparison between normalized PSTH rasters for one target');

[latent_A,~,~,~] = get_pca(A_norm,3);
[latent_N,~,~,~] = get_pca(N_norm,3);

PSTH_aux = obtain_PSTH(latent_A,new_targets,index_test);
PSTH_N_aux = obtain_PSTH(latent_N,new_targets,index_test);

figure;
subplot(1,2,1);
plot3(squeeze(PSTH_aux(1,1,:)),squeeze(PSTH_aux(2,1,:)),squeeze(PSTH_aux(3,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot3(squeeze(PSTH_aux(1,2,:)),squeeze(PSTH_aux(2,2,:)),squeeze(PSTH_aux(3,2,:)),'Color',[0 1 0],'LineWidth',3);
plot3(squeeze(PSTH_aux(1,3,:)),squeeze(PSTH_aux(2,3,:)),squeeze(PSTH_aux(3,3,:)),'Color',[0 0 1],'LineWidth',3);
plot3(squeeze(PSTH_aux(1,4,:)),squeeze(PSTH_aux(2,4,:)),squeeze(PSTH_aux(3,4,:)),'Color',[1 1 0],'LineWidth',3);
plot3(squeeze(PSTH_aux(1,5,:)),squeeze(PSTH_aux(2,5,:)),squeeze(PSTH_aux(3,5,:)),'Color',[1 0 1],'LineWidth',3);
plot3(squeeze(PSTH_aux(1,6,:)),squeeze(PSTH_aux(2,6,:)),squeeze(PSTH_aux(3,6,:)),'Color',[0 1 1],'LineWidth',3);
plot3(squeeze(PSTH_aux(1,7,:)),squeeze(PSTH_aux(2,7,:)),squeeze(PSTH_aux(3,7,:)),'Color',[0.7 0.7 0.7],'LineWidth',3);
plot3(squeeze(PSTH_aux(1,8,:)),squeeze(PSTH_aux(2,8,:)),squeeze(PSTH_aux(3,8,:)),'Color',[0 0 0],'LineWidth',3);
subplot(1,2,2);
plot3(squeeze(PSTH_N_aux(1,1,:)),squeeze(PSTH_N_aux(2,1,:)),squeeze(PSTH_N_aux(3,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot3(squeeze(PSTH_N_aux(1,2,:)),squeeze(PSTH_N_aux(2,2,:)),squeeze(PSTH_N_aux(3,2,:)),'Color',[0 1 0],'LineWidth',3);
plot3(squeeze(PSTH_N_aux(1,3,:)),squeeze(PSTH_N_aux(2,3,:)),squeeze(PSTH_N_aux(3,3,:)),'Color',[0 0 1],'LineWidth',3);
plot3(squeeze(PSTH_N_aux(1,4,:)),squeeze(PSTH_N_aux(2,4,:)),squeeze(PSTH_N_aux(3,4,:)),'Color',[1 1 0],'LineWidth',3);
plot3(squeeze(PSTH_N_aux(1,5,:)),squeeze(PSTH_N_aux(2,5,:)),squeeze(PSTH_N_aux(3,5,:)),'Color',[1 0 1],'LineWidth',3);
plot3(squeeze(PSTH_N_aux(1,6,:)),squeeze(PSTH_N_aux(2,6,:)),squeeze(PSTH_N_aux(3,6,:)),'Color',[0 1 1],'LineWidth',3);
plot3(squeeze(PSTH_N_aux(1,7,:)),squeeze(PSTH_N_aux(2,7,:)),squeeze(PSTH_N_aux(3,7,:)),'Color',[0.7 0.7 0.7],'LineWidth',3);
plot3(squeeze(PSTH_N_aux(1,8,:)),squeeze(PSTH_N_aux(2,8,:)),squeeze(PSTH_N_aux(3,8,:)),'Color',[0 0 0],'LineWidth',3);

figure;
subplot(1,2,1);
plot(squeeze(PSTH_aux(1,1,:)),squeeze(PSTH_aux(2,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot(squeeze(PSTH_aux(1,2,:)),squeeze(PSTH_aux(2,2,:)),'Color',[0 1 0],'LineWidth',3);
plot(squeeze(PSTH_aux(1,3,:)),squeeze(PSTH_aux(2,3,:)),'Color',[0 0 1],'LineWidth',3);
plot(squeeze(PSTH_aux(1,4,:)),squeeze(PSTH_aux(2,4,:)),'Color',[1 1 0],'LineWidth',3);
plot(squeeze(PSTH_aux(1,5,:)),squeeze(PSTH_aux(2,5,:)),'Color',[1 0 1],'LineWidth',3);
plot(squeeze(PSTH_aux(1,6,:)),squeeze(PSTH_aux(2,6,:)),'Color',[0 1 1],'LineWidth',3);
plot(squeeze(PSTH_aux(1,7,:)),squeeze(PSTH_aux(2,7,:)),'Color',[0.7 0.7 0.7],'LineWidth',3);
plot(squeeze(PSTH_aux(1,8,:)),squeeze(PSTH_aux(2,8,:)),'Color',[0 0 0],'LineWidth',3);
subplot(1,2,2);
plot(squeeze(PSTH_N_aux(1,1,:)),squeeze(PSTH_N_aux(2,1,:)),'Color',[1 0 0],'LineWidth',3);
hold on;
plot(squeeze(PSTH_N_aux(1,2,:)),squeeze(PSTH_N_aux(2,2,:)),'Color',[0 1 0],'LineWidth',3);
plot(squeeze(PSTH_N_aux(1,3,:)),squeeze(PSTH_N_aux(2,3,:)),'Color',[0 0 1],'LineWidth',3);
plot(squeeze(PSTH_N_aux(1,4,:)),squeeze(PSTH_N_aux(2,4,:)),'Color',[1 1 0],'LineWidth',3);
plot(squeeze(PSTH_N_aux(1,5,:)),squeeze(PSTH_N_aux(2,5,:)),'Color',[1 0 1],'LineWidth',3);
plot(squeeze(PSTH_N_aux(1,6,:)),squeeze(PSTH_N_aux(2,6,:)),'Color',[0 1 1],'LineWidth',3);
plot(squeeze(PSTH_N_aux(1,7,:)),squeeze(PSTH_N_aux(2,7,:)),'Color',[0.7 0.7 0.7],'LineWidth',3);
plot(squeeze(PSTH_N_aux(1,8,:)),squeeze(PSTH_N_aux(2,8,:)),'Color',[0 0 0],'LineWidth',3);

%% dPSTHs:

dPSTH = zeros(size(PSTH(:,:,1:end-1)));
dPSTH_N = zeros(size(PSTH_N(:,:,1:end-1)));
for i = 2:time
    dPSTH(:,:,i) = PSTH(:,:,i) - PSTH(:,:,i-1);
    dPSTH_N(:,:,i) = PSTH_N(:,:,i) - PSTH_N(:,:,i-1);
end

figure;
plot(squeeze(mean(mean(abs(dPSTH),1),2)),'Color',[0 0.5 1],'LineWidth',2);
hold on;
plot(squeeze(mean(mean(abs(dPSTH_N),1),2)),'Color',[0 1 0.5], ...
    'LineWidth',2);
xlabel('time');
ylabel('dPSTH');
xline(t-13,'k','LineWidth',3);
xlim([2 time]);
h = legend({'RNN units','Neurons'},'Location','SouthOutside');
set(h,'Box','off');
title('Mean of absolute value of dPSTHs for RNN units and neurons');

%% Activations:

time = size(N_test,3);

% Activations of a RNN unit in all test trials with targets 1, 2 and 3:

unit = 16;

figure;
subplot(1,2,1);
aux = zeros(3,time);
counter = zeros(1,3);
color = {'blue' 'red' 'green'};
for i = 1:size(A_norm,2)
    for j = 1:3
        if new_targets(index_test(i)) == j && counter(j) <= 5
            plot(squeeze(A_norm(unit,i,:)),color{(j)},'LineWidth',0.4);
            hold on;
            aux(j,:) = aux(j,:) + transpose(squeeze(A_norm(unit,i,:)));
            counter(j) = counter(j) + 1;
        end
    end
end
aux(1,:) = aux(1,:)/counter(1);
aux(2,:) = aux(2,:)/counter(2);
aux(3,:) = aux(3,:)/counter(3);
plot(aux(1,:),'Color',[0 0 0.6],'LineWidth',4);
hold on;
plot(aux(2,:),'Color',[0.6 0 0],'LineWidth',4);
plot(aux(3,:),'Color',[0 0.6 0],'LineWidth',4);
xline(t-14,'k','LineWidth',2);
axis tight;
xlabel('time');
ylabel('rate of firing');
title('Activations of a RNN unit in different trials with targets 1, 2 and 3');

% Activations of neuron 1 in all test trials with targets 1, 2 and 3:

neuron = 2;

subplot(1,2,2);
aux = zeros(3,time);
counter = zeros(1,3);
color = {'blue' 'red' 'green'};
for i = 1:size(N_norm,2)
    for j = 1:3
        if new_targets(index_test(i)) == j && counter(j) <= 5
            plot(squeeze(N_norm(neuron,i,:)),color{(j)},'LineWidth',0.4);
            hold on;
            aux(j,:) = aux(j,:) + transpose(squeeze(N_norm(neuron,i,:)));
            counter(j) = counter(j) + 1;
        end
    end
end
aux(1,:) = aux(1,:)/counter(1);
aux(2,:) = aux(2,:)/counter(2);
aux(3,:) = aux(3,:)/counter(3);
plot(aux(1,:),'Color',[0 0 0.6],'LineWidth',4);
hold on;
plot(aux(2,:),'Color',[0.6 0 0],'LineWidth',4);
plot(aux(3,:),'Color',[0 0.6 0],'LineWidth',4);
xline(t-14,'k','LineWidth',2);
axis tight;
xlabel('time');
ylabel('rate of firing');
title('Activations of a neuron in different trials with targets 1, 2 and 3');

%% Graph of connections between units:

grph = graph(net.G(:,1:param.n_hidden));

figure;
plot(grph,'NodeLabel',{},'Marker','o','MarkerSize',3,'Layout','auto', ...
    'LineWidth',1,'NodeColor',[0 0 0.8],'EdgeColor',[0 0.5 1]);
title('Graph of connections between units');

%% Preparatory activity of RNN units (PCA):

% 2D:

dims_pca = 2;

color = {[1,0,0] [0,1,0] [0,0,1] ...
    [1,1,0] [1,0,1] [0,1,1] ...
    [0.7,0.7,0.7] [0,0,0]};

[A_pca,~,~] = get_pca(A_test,dims_pca);
A_pca_prep = A_pca(:,:,t-20:t-14);
A_pca_sum = sum(A_pca_prep,3);

figure;
subplot(1,2,1)
for i = 1:size(A_test,2)
    plot(A_pca_sum(1,i),A_pca_sum(2,i),'.', ...
        'Color',color{(new_targets(index_test(i)))},'MarkerSize',20);
    hold on;
end
xlabel('PC 1');
ylabel('PC 2');
title('RNN units');

subplot(1,2,2)
for i = 1:size(A_test,2)
    plot(mean(X_test(1,i,end-simulation_bins(2)-8:end-simulation_bins(2)-1),'all'), ...
        mean(X_test(2,i,end-simulation_bins(2)-8:end-simulation_bins(2)-1),'all'), ...
        '.','Color',color{(new_targets(index_test(i)))},'MarkerSize',20);
    hold on;
end
for i = 1:8
    plot(mean(mean(X_test(1,new_targets(index_test) == i,end-simulation_bins(2)-8:end-simulation_bins(2)-1),3),2), ...
        mean(mean(X_test(2,new_targets(index_test) == i,end-simulation_bins(2)-8:end-simulation_bins(2)-1),3),2), ...
        '.','Color',color{(i)},'MarkerSize',70);
end
xlabel('PC 1');
ylabel('PC 2');
title('Neurons');

suptitle('Target classification with the first two PC of preparatory activity');

% 3D:

dims_pca = 3;
[A_pca,~,~] = get_pca(A_test,dims_pca);
A_pca_prep = A_pca(:,:,t-20:t-14);
A_pca_sum = sum(A_pca_prep,3);

figure;
subplot(1,2,1)
for i = 1:size(A_test,2)
    plot3(A_pca_sum(1,i),A_pca_sum(2,i),A_pca_sum(3,i),'.', ...
        'Color',color{(new_targets(index_test(i)))},'MarkerSize',20);
    hold on;
end
xlabel('PC 1');
ylabel('PC 2');
title('RNN units');

subplot(1,2,2)
for i = 1:size(A_test,2)
    plot3(X_test(1,i,end-simulation_bins(2)-1), ...
        X_test(2,i,end-simulation_bins(2)-1), ...
        X_test(3,i,end-simulation_bins(2)-1),'.', ...
        'Color',color{(new_targets(index_test(i)))},'MarkerSize',20);
    hold on;
end
xlabel('PC 1');
ylabel('PC 2');
title('Neurons');

suptitle('Target classification with the first three PC of preparatory activity');

%% Shuffling, smoothing and Procrustes / CCA: neurons vs units

% Time and targets:

NN = zeros(size(N_test,1),size(N_test,2)*size(N_test,3));
AA = zeros(size(A_test,1),size(A_test,2)*size(A_test,3));
for j = 1:size(N_test,2)
        NN(:,(size(N_test,3)*(j-1)+1):(size(N_test,3)*j)) = ...
            squeeze(N_test(:,j,:));
        AA(:,(size(A_test,3)*(j-1)+1):(size(A_test,3)*j)) = ...
            squeeze(A_test(:,j,:));
end

idx = randperm(size(NN,2));
NN_shuffle = NN;
NN_shuffle(:,idx) = NN; 
AA_shuffle = AA;
AA_shuffle(:,idx) = AA; 

% Time and targets:

width_smooth_shuffle = 0.1;
NN_shuffle_smooth = smooth_data(transpose(NN_shuffle),td(1).bin_size, ...
    width_smooth_shuffle);
NN_shuffle_smooth = transpose(NN_shuffle_smooth);
AA_shuffle_smooth = smooth_data(transpose(AA_shuffle),td(1).bin_size, ...
    width_smooth_shuffle);
AA_shuffle_smooth = transpose(AA_shuffle_smooth);

NN_ft = fft(NN(1,:));
NN_shuffle_ft = fft(NN_shuffle(1,:));
NN_shuffle_smooth_ft = fft(NN_shuffle_smooth(1,:));

AA_ft = fft(AA(1,:));
AA_shuffle_ft = fft(AA_shuffle(1,:));
AA_shuffle_smooth_ft = fft(AA_shuffle_smooth(1,:));

T = td(1).bin_size;
Fs = 1/T;
L = size(NN,2);
t = (0:L-1)*T;
f = Fs*(0:(L/2))/L;

P2_NN = abs(NN_ft/L);
P1_NN = P2_NN(1:L/2+1);
P1_NN(2:end-1) = 2*P1_NN(2:end-1);
P2_NN_shuffle = abs(NN_shuffle_ft/L);
P1_NN_shuffle = P2_NN_shuffle(1:L/2+1);
P1_NN_shuffle(2:end-1) = 2*P1_NN_shuffle(2:end-1);
P2_NN_shuffle_smooth = abs(NN_shuffle_smooth_ft/L);
P1_NN_shuffle_smooth = P2_NN_shuffle_smooth(1:L/2+1);
P1_NN_shuffle_smooth(2:end-1) = 2*P1_NN_shuffle_smooth(2:end-1);

P2_AA = abs(AA_ft/L);
P1_AA = P2_AA(1:L/2+1);
P1_AA(2:end-1) = 2*P1_AA(2:end-1);
P2_AA_shuffle = abs(AA_shuffle_ft/L);
P1_AA_shuffle = P2_AA_shuffle(1:L/2+1);
P1_AA_shuffle(2:end-1) = 2*P1_AA_shuffle(2:end-1);
P2_AA_shuffle_smooth = abs(AA_shuffle_smooth_ft/L);
P1_AA_shuffle_smooth = P2_AA_shuffle_smooth(1:L/2+1);
P1_AA_shuffle_smooth(2:end-1) = 2*P1_AA_shuffle_smooth(2:end-1);

% Targets:

NN_shuffle_targets = zeros(size(NN));
AA_shuffle_targets = zeros(size(AA));
idx = randperm(size(N_test,2));
for i = 1:size(N_test,2)
    NN_shuffle_targets(:,(size(N_test,3)*(i-1)+1):(size(N_test,3)*i)) = ...
        squeeze(N_test(:,idx(i),:));
    AA_shuffle_targets(:,(size(A_test,3)*(i-1)+1):(size(A_test,3)*i)) = ...
        squeeze(A_test(:,idx(i),:));
end

% Time:

idx = randperm(size(N_test,3)) ;
N_shuffle_time = N_test;
N_shuffle_time(:,:,idx) = N_test; 
A_shuffle_time = A_test;
A_shuffle_time(:,:,idx) = A_test; 

NN_shuffle_time = zeros(size(N_test,1),size(N_test,2)*size(N_test,3));
AA_shuffle_time = zeros(size(A_test,1),size(A_test,2)*size(A_test,3));
for i = 1:size(N_test,2)
    NN_shuffle_time(:,(size(N_test,3)*(i-1)+1):(size(N_test,3)*i)) = ...
        squeeze(N_shuffle_time(:,i,:));
    AA_shuffle_time(:,(size(A_test,3)*(i-1)+1):(size(A_test,3)*i)) = ...
        squeeze(A_shuffle_time(:,i,:));
end

NN_shuffle_time_smooth = smooth_data(transpose(NN_shuffle_time), ...
    td(1).bin_size,width_smooth_shuffle);
NN_shuffle_time_smooth = transpose(NN_shuffle_time_smooth);
AA_shuffle_time_smooth = smooth_data(transpose(AA_shuffle_time), ...
    td(1).bin_size,width_smooth_shuffle);
AA_shuffle_time_smooth = transpose(AA_shuffle_time_smooth);

NN_ft = fft(NN(1,:));
NN_shuffle_ft = fft(NN_shuffle_time(1,:));
NN_shuffle_smooth_ft = fft(NN_shuffle_time_smooth(1,:));

AA_ft = fft(AA(1,:));
AA_shuffle_ft = fft(AA_shuffle_time(1,:));
AA_shuffle_smooth_ft = fft(AA_shuffle_time_smooth(1,:));

T = td(1).bin_size;
Fs = 1/T;
L = size(NN,2);
t = (0:L-1)*T;
f = Fs*(0:(L/2))/L;

P2_NN = abs(NN_ft/L);
P1_NN = P2_NN(1:L/2+1);
P1_NN(2:end-1) = 2*P1_NN(2:end-1);
P2_NN_shuffle = abs(NN_shuffle_ft/L);
P1_NN_shuffle = P2_NN_shuffle(1:L/2+1);
P1_NN_shuffle(2:end-1) = 2*P1_NN_shuffle(2:end-1);
P2_NN_shuffle_smooth = abs(NN_shuffle_smooth_ft/L);
P1_NN_shuffle_smooth = P2_NN_shuffle_smooth(1:L/2+1);
P1_NN_shuffle_smooth(2:end-1) = 2*P1_NN_shuffle_smooth(2:end-1);

P2_AA = abs(AA_ft/L);
P1_AA = P2_AA(1:L/2+1);
P1_AA(2:end-1) = 2*P1_AA(2:end-1);
P2_AA_shuffle = abs(AA_shuffle_ft/L);
P1_AA_shuffle = P2_AA_shuffle(1:L/2+1);
P1_AA_shuffle(2:end-1) = 2*P1_AA_shuffle(2:end-1);
P2_AA_shuffle_smooth = abs(AA_shuffle_smooth_ft/L);
P1_AA_shuffle_smooth = P2_AA_shuffle_smooth(1:L/2+1);
P1_AA_shuffle_smooth(2:end-1) = 2*P1_AA_shuffle_smooth(2:end-1);

dims = 15;
[AA_pca,~,~] = get_pca(AA,dims);
[NN_pca,~,~] = get_pca(NN,dims);

[NN_shuffle_time_smooth_pca,~,~] = get_pca(NN_shuffle_time_smooth,dims);
[NN_shuffle_time_pca,~,~] = get_pca(NN_shuffle_time,dims);
[NN_shuffle_targets_pca,~,~] = get_pca(NN_shuffle_targets,dims);
[NN_shuffle_smooth_pca,~,~] = get_pca(NN_shuffle_smooth,dims);
[NN_shuffle_pca,~,~] = get_pca(NN_shuffle,dims);

[AA_shuffle_time_smooth_pca,~,~] = get_pca(AA_shuffle_time_smooth,dims);
[AA_shuffle_time_pca,~,~] = get_pca(AA_shuffle_time,dims);
[AA_shuffle_targets_pca,~,~] = get_pca(AA_shuffle_targets,dims);
[AA_shuffle_smooth_pca,~,~] = get_pca(AA_shuffle_smooth,dims);
[AA_shuffle_pca,~,~] = get_pca(AA_shuffle,dims);

[~,~,r_1,~,~] = canoncorr(transpose(NN_pca),transpose(AA_pca));
d_1 = procrustes(transpose(NN_pca),transpose(AA_pca));

[~,~,r_2,~,~] = canoncorr(transpose(NN_shuffle_time_smooth_pca),transpose(AA_pca));
d_2 = procrustes(transpose(NN_shuffle_time_smooth_pca),transpose(AA_pca));
[~,~,r_3,~,~] = canoncorr(transpose(NN_shuffle_time_pca),transpose(AA_pca));
d_3 = procrustes(transpose(NN_shuffle_time_pca),transpose(AA_pca));
[~,~,r_4,~,~] = canoncorr(transpose(NN_shuffle_targets_pca),transpose(AA_pca));
d_4 = procrustes(transpose(NN_shuffle_targets_pca),transpose(AA_pca));
[~,~,r_5,~,~] = canoncorr(transpose(NN_shuffle_smooth_pca),transpose(AA_pca));
d_5 = procrustes(transpose(NN_shuffle_smooth_pca),transpose(AA_pca));
[~,~,r_6,~,~] = canoncorr(transpose(NN_shuffle_pca),transpose(AA_pca));
d_6 = procrustes(transpose(NN_shuffle_pca),transpose(AA_pca));

[~,~,r_7,~,~] = canoncorr(transpose(NN_pca),transpose(AA_shuffle_time_smooth_pca));
d_7 = procrustes(transpose(NN_pca),transpose(AA_shuffle_time_smooth_pca));
[~,~,r_8,~,~] = canoncorr(transpose(NN_pca),transpose(AA_shuffle_time_pca));
d_8 = procrustes(transpose(NN_pca),transpose(AA_shuffle_time_pca));
[~,~,r_9,~,~] = canoncorr(transpose(NN_pca),transpose(AA_shuffle_targets_pca));
d_9 = procrustes(transpose(NN_pca),transpose(AA_shuffle_targets_pca));
[~,~,r_10,~,~] = canoncorr(transpose(NN_pca),transpose(AA_shuffle_smooth_pca));
d_10 = procrustes(transpose(NN_pca),transpose(AA_shuffle_smooth_pca));
[~,~,r_11,~,~] = canoncorr(transpose(NN_pca),transpose(AA_shuffle_pca));
d_11 = procrustes(transpose(NN_pca),transpose(AA_shuffle_pca));

figure;
subplot(1,2,1);
plot(r_1,'Color',[0 0.5 1],'LineWidth',2);
hold on;
plot(r_2,'Color',[1 0.7 0],'LineWidth',2);
plot(r_3,'Color',[0.8 0.5 0],'LineWidth',1,'LineStyle','--');
plot(r_4,'Color',[0 1 0.5],'LineWidth',2);
plot(r_5,'Color',[1 0 0],'LineWidth',2);
plot(r_6,'Color',[0.8 0 0],'LineWidth',1,'LineStyle','--');
title('Canonical correlations (CCA) of latent dynamics: units vs neurons');
h = legend({'No distortion in neurons', ...
    'Time distortion in neurons (smoothed)', ...
    'Time distortion in neurons', ...
    'Target distortion in neurons', ...
    'Time and target distortion in neurons (smoothed)', ...
    'Time and target distortion in neurons'}, ...
    'Location','SouthOutside');
set(h,'Box','off');
axis tight;
ylim([0 1]);

subplot(1,2,2);
plot(r_1,'Color',[0 0.5 1],'LineWidth',2);
hold on;
plot(r_7,'Color',[1 0.7 0],'LineWidth',2);
plot(r_8,'Color',[0.8 0.5 0],'LineWidth',1,'LineStyle','--');
plot(r_9,'Color',[0 1 0.5],'LineWidth',2);
plot(r_10,'Color',[1 0 0],'LineWidth',2);
plot(r_11,'Color',[0.8 0 0],'LineWidth',1,'LineStyle','--');
title('Canonical correlations (CCA) of latent dynamics: units vs neurons');
h = legend({'No distortion in units', ...
    'Time distortion in units (smoothed)', ...
    'Time distortion in units', ...
    'Target distortion in units', ...
    'Time and target distortion in units (smoothed)', ...
    'Time and target distortion in units'}, ...
    'Location','SouthOutside');
set(h,'Box','off');
axis tight;
ylim([0 1]);

figure;
x = categorical({'Neurons','Units'});
y = [d_1 d_2 d_3 d_4 d_5 d_6; d_1 d_7 d_8 d_9 d_10 d_11];
b = bar(x,y);
b(1).FaceColor = [0 0.5 1];
b(1).EdgeColor = [0 0.5 1];
b(2).FaceColor = [1 0.7 0];
b(2).EdgeColor = [1 0.7 0];
b(3).FaceColor = [0.8 0.5 0];
b(3).EdgeColor = [0.8 0.5 0];
b(4).FaceColor = [0 1 0.5];
b(4).EdgeColor = [0 1 0.5];
b(5).FaceColor = [1 0 0];
b(5).EdgeColor = [1 0 0];
b(6).FaceColor = [0.8 0 0];
b(6).EdgeColor = [0.8 0 0];
title('Procrustes analysis (distortion in...)');
ylabel('procrustes dissimilarity');
h = legend({'No','Time (smoothed)','Time','Targets', ...
    'Time + targets (smoothed)','Time + targets'},'Location', ...
    'southoutside');
set(h,'Box','off');

%% Trained vs untrained:

load('trained_vs_untrained.mat');

d_trained = compare_layers_procrustes(A_trained,N_data,10);
d_untrained = compare_layers_procrustes(A_untrained,N_data,10);

figure;
x = categorical({'Units'});
y = [d_trained d_untrained];
b = bar(x,y);
b(1).FaceColor = [0 0.5 1];
b(1).EdgeColor = [0 0.5 1];
b(2).FaceColor = [0 1 0.5];
b(2).EdgeColor = [0 1 0.5];
title('Procrustes analysis with neurons');
ylabel('procrustes dissimilarity');
h = legend({'Trained','Untrained'});
set(h,'Box','off');

%% Weight analysis:

l_1 = sum(net_100.G(1:size(net_100.G,1),1:size(net_100.G,1)), ...
    'all');
l_2 = sum(net_1.G(1:size(net_1.G,1),1:size(net_1.G,1)), ...
    'all');
l_3 = sum(net_10.G(1:size(net_10.G,1),1:size(net_10.G,1)), ...
    'all');

W_f_1 = zeros(1,l_1);
W_i_1 = zeros(1,l_1);
W_c_1 = zeros(1,l_1);
W_o_1 = zeros(1,l_1);
W_f_init_1 = zeros(1,l_1);
W_i_init_1 = zeros(1,l_1);
W_c_init_1 = zeros(1,l_1);
W_o_init_1 = zeros(1,l_1);
counter = 0;
for i = 1:size(net_100.G,1)
    for j = 1:size(net_100.G,1)
        if net_100.G(i,j) == 1
            counter = counter + 1;
            W_f_1(counter) = net_100.W_f(i,j);
            W_i_1(counter) = net_100.W_i(i,j);
            W_c_1(counter) = net_100.W_c(i,j);
            W_o_1(counter) = net_100.W_o(i,j);
            W_f_init_1(counter) = net_100.init.W_f(i,j);
            W_i_init_1(counter) = net_100.init.W_i(i,j);
            W_c_init_1(counter) = net_100.init.W_c(i,j);
            W_o_init_1(counter) = net_100.init.W_o(i,j);
        end
    end
end
W_f_2 = zeros(1,l_2);
W_i_2 = zeros(1,l_2);
W_c_2 = zeros(1,l_2);
W_o_2 = zeros(1,l_2);
W_f_init_2 = zeros(1,l_2);
W_i_init_2 = zeros(1,l_2);
W_c_init_2 = zeros(1,l_2);
W_o_init_2 = zeros(1,l_2);
counter = 0;
for i = 1:size(net_1.G,1)
    for j = 1:size(net_1.G,1)
        if net_1.G(i,j) == 1
            counter = counter + 1;
            W_f_2(counter) = net_1.W_f(i,j);
            W_i_2(counter) = net_1.W_i(i,j);
            W_c_2(counter) = net_1.W_c(i,j);
            W_o_2(counter) = net_1.W_o(i,j);
            W_f_init_2(counter) = net_1.init.W_f(i,j);
            W_i_init_2(counter) = net_1.init.W_i(i,j);
            W_c_init_2(counter) = net_1.init.W_c(i,j);
            W_o_init_2(counter) = net_1.init.W_o(i,j);
        end
    end
end
W_f_3 = zeros(1,l_3);
W_i_3 = zeros(1,l_3);
W_c_3 = zeros(1,l_3);
W_o_3 = zeros(1,l_3);
W_f_init_3 = zeros(1,l_3);
W_i_init_3 = zeros(1,l_3);
W_c_init_3 = zeros(1,l_3);
W_o_init_3 = zeros(1,l_3);
counter = 0;
for i = 1:size(net_10.G,1)
    for j = 1:size(net_10.G,1)
        if net_10.G(i,j) == 1
            counter = counter + 1;
            W_f_3(counter) = net_10.W_f(i,j);
            W_i_3(counter) = net_10.W_i(i,j);
            W_c_3(counter) = net_10.W_c(i,j);
            W_o_3(counter) = net_10.W_o(i,j);
            W_f_init_3(counter) = net_10.init.W_f(i,j);
            W_i_init_3(counter) = net_10.init.W_i(i,j);
            W_c_init_3(counter) = net_10.init.W_c(i,j);
            W_o_init_3(counter) = net_10.init.W_o(i,j);
        end
    end
end

bins = 30;

figure;

subplot(1,3,1);
h1 = histfit(W_o_2,bins,'kernel');
hold on
h2 = histfit(W_o_init_2,bins,'kernel');
h1(1).FaceColor = [0 1 0.6];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 0.7 0];
h2(1).FaceColor = [0 0.6 1];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 0 0.7];
title('1 % connectivity');
xlabel('weight value');
xlim([-1.3 1.3]);
ylim([0 150]);
h = legend({'Post-training','Post-training fit','Pre-training', ...
    'Pre-training fit'},'Location','northeast');
set(h,'Box','off');

subplot(1,3,2);
h1 = histfit(W_o_3,bins,'kernel');
hold on
h2 = histfit(W_o_init_3,bins,'kernel');
h1(1).FaceColor = [0 1 0.6];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 0.7 0];
h2(1).FaceColor = [0 0.6 1];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 0 0.7];
title('10 % connectivity');
xlabel('weight value');
xlim([-1.3 1.3]);
ylim([0 1800]);
h = legend({'Post-training','Post-training fit','Pre-training', ...
    'Pre-training fit'},'Location','northeast');
set(h,'Box','off');

subplot(1,3,3);
h1 = histfit(W_o_1,bins,'kernel');
hold on
h2 = histfit(W_o_init_1,bins,'kernel');
h1(1).FaceColor = [0 1 0.6];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 0.7 0];
h2(1).FaceColor = [0 0.6 1];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 0 0.7];
title('100 % connectivity');
xlabel('weight value');
xlim([-1.3 1.3]);
ylim([0 27000]);
h = legend({'Post-training','Post-training fit','Pre-training', ...
    'Pre-training fit'},'Location','northeast');
set(h,'Box','off');

%% Correlation damping:

num_iter = 50;
dims = 40;
C_A_no_corr_damping = ...
    obtain_correlation_matrix(A_no_corr_damping,dims,num_iter,'no');
C_A_corr_damping = ...
    obtain_correlation_matrix(A_corr_damping,dims,num_iter,'no');

figure;
subplot(1,2,1);
image(abs(C_A_no_corr_damping)*500);
title('No damping');
subplot(1,2,2);
image(abs(C_A_corr_damping)*500);
title('Damping');
suptitle('Correlation matrices');

vec_no_corr_damping = [];
vec_corr_damping = [];
for i = 1:dims
    for j = 1:dims
        if i ~= j
            vec_no_corr_damping = [vec_no_corr_damping C_A_no_corr_damping(i,j)];
            vec_corr_damping = [vec_corr_damping C_A_corr_damping(i,j)];
        end
    end
end

figure;
h1 = histfit(abs(vec_no_corr_damping),20,'kernel');
hold on
h2 = histfit(abs(vec_corr_damping),20,'kernel');
h1(1).FaceColor = [0 1 0.6];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 0.7 0];
h2(1).FaceColor = [0 0.6 1];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 0 0.7];
h = legend({'No damping','No damping fit','Damping','Damping fit'}, ...
    'Location','northeast');
set(h,'Box','off');
title('Correlations');

%% GENERAL ANALYSIS: NEURONS

load('New_neural_activity.mat');

trials = 1:50;
sp = 5;
iter = 3;

N = N_C_1_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C_1 = num;
f_C_1 = fit(transpose(num),transpose(r),'power2');
f_C_1 = f_C_1(sp:num(end));

N = N_C_2_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C_2 = num;
f_C_2 = fit(transpose(num),transpose(r),'power2');
f_C_2 = f_C_2(sp:num(end));

N = N_C_3_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C_3 = num;
f_C_3 = fit(transpose(num),transpose(r),'power2');
f_C_3 = f_C_3(sp:num(end));

N = N_C_4_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C_4 = num;
f_C_4 = fit(transpose(num),transpose(r),'power2');
f_C_4 = f_C_4(sp:num(end));

N = N_C2_1_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C2_1 = num;
f_C2_1 = fit(transpose(num),transpose(r),'power2');
f_C2_1 = f_C2_1(sp:num(end));

N = N_C2_2_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C2_2 = num;
f_C2_2 = fit(transpose(num),transpose(r),'power2'); 
f_C2_2 = f_C2_2(sp:num(end));

N = N_C2_3_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C2_3 = num;
f_C2_3 = fit(transpose(num),transpose(r),'power2');
f_C2_3 = f_C2_3(sp:num(end));

N = N_C2_4_M1; 
[r,num] = calculate_ratios(N,iter,trials,sp);
num_C2_4 = num;
f_C2_4 = fit(transpose(num),transpose(r),'power2');
f_C2_4 = f_C2_4(sp:num(end));

figure;
plot(sp:num_C_1(end),f_C_1,'Color',[0 0 1],'LineWidth',3);
hold on;
plot(sp:num_C_2(end),f_C_2,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C_3(end),f_C_3,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C_4(end),f_C_4,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C2_1(end),f_C2_1,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C2_2(end),f_C2_2,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C2_3(end),f_C2_3,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C2_4(end),f_C2_4,'Color',[0 0 1],'LineWidth',3);
plot(sp:num_C2_3(end),ones(1,num_C2_3(end)-sp+1),'k--');
axis tight;
ylim([0.5 1.1]);
title('Ratio ISOMAP / PCA (est. dims.)');
xlabel('number of neurons');
ylabel('ratio');

iter = 10;

N = N_C_1_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C_1 = m_i;
exp_p_C_1 = m_p;

N = N_C_2_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C_2 = m_i;
exp_p_C_2 = m_p;

N = N_C_3_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C_3 = m_i;
exp_p_C_3 = m_p;

N = N_C_4_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C_4 = m_i;
exp_p_C_4 = m_p;

N = N_C2_1_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C2_1 = m_i;
exp_p_C2_1 = m_p;

N = N_C2_2_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C2_2 = m_i;
exp_p_C2_2 = m_p;

N = N_C2_3_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C2_3 = m_i;
exp_p_C2_3 = m_p;

N = N_C2_4_M1; 
[m_i,~,m_p,~] = compare_performance(N,iter,size(N,1),trials);
exp_i_C2_4 = m_i;
exp_p_C2_4 = m_p;

rat_C_1 = exp_i_C_1./exp_p_C_1;
rat_C_2 = exp_i_C_2./exp_p_C_2;
rat_C_3 = exp_i_C_3./exp_p_C_3;
rat_C_4 = exp_i_C_4./exp_p_C_4;
rat_C2_1 = exp_i_C2_1./exp_p_C2_1;
rat_C2_2 = exp_i_C2_2./exp_p_C2_2;
rat_C2_3 = exp_i_C2_3./exp_p_C2_3;
rat_C2_4 = exp_i_C2_4./exp_p_C2_4;

%{
figure;
plot(rat_C_1,'Color',[1 0 0],'LineWidth',3);
hold on;
plot(rat_C_2,'Color',[1 0 0],'LineWidth',3);
plot(rat_C_3,'Color',[1 0 0],'LineWidth',3);
plot(rat_C_4,'Color',[1 0 0],'LineWidth',3);
plot(rat_C2_1,'Color',[1 0 0],'LineWidth',3);
plot(rat_C2_2,'Color',[1 0 0],'LineWidth',3);
plot(rat_C2_3,'Color',[1 0 0],'LineWidth',3);
plot(rat_C2_4,'Color',[1 0 0],'LineWidth',3);
plot(1:77,ones(1,77),'k--');
axis tight;
ylim([0.9 1.5]);
xlim([1 77]);
title('Ratio ISOMAP / PCA (exp. var.)');
xlabel('dimensions');
ylabel('ratio');
%}

exp_i_C_1_norm = interp1(linspace(1,100,length(exp_i_C_1)),exp_i_C_1,linspace(1,100,100));
exp_i_C_2_norm = interp1(linspace(1,100,length(exp_i_C_2)),exp_i_C_2,linspace(1,100,100));
exp_i_C_3_norm = interp1(linspace(1,100,length(exp_i_C_3)),exp_i_C_3,linspace(1,100,100));
exp_i_C_4_norm = interp1(linspace(1,100,length(exp_i_C_4)),exp_i_C_4,linspace(1,100,100));
exp_i_C2_1_norm = interp1(linspace(1,100,length(exp_i_C2_1)),exp_i_C2_1,linspace(1,100,100));
exp_i_C2_2_norm = interp1(linspace(1,100,length(exp_i_C2_2)),exp_i_C2_2,linspace(1,100,100));
exp_i_C2_3_norm = interp1(linspace(1,100,length(exp_i_C2_3)),exp_i_C2_3,linspace(1,100,100));
exp_i_C2_4_norm = interp1(linspace(1,100,length(exp_i_C2_4)),exp_i_C2_4,linspace(1,100,100));

exp_p_C_1_norm = interp1(linspace(1,100,length(exp_p_C_1)),exp_p_C_1,linspace(1,100,100));
exp_p_C_2_norm = interp1(linspace(1,100,length(exp_p_C_2)),exp_p_C_2,linspace(1,100,100));
exp_p_C_3_norm = interp1(linspace(1,100,length(exp_p_C_3)),exp_p_C_3,linspace(1,100,100));
exp_p_C_4_norm = interp1(linspace(1,100,length(exp_p_C_4)),exp_p_C_4,linspace(1,100,100));
exp_p_C2_1_norm = interp1(linspace(1,100,length(exp_p_C2_1)),exp_p_C2_1,linspace(1,100,100));
exp_p_C2_2_norm = interp1(linspace(1,100,length(exp_p_C2_2)),exp_p_C2_2,linspace(1,100,100));
exp_p_C2_3_norm = interp1(linspace(1,100,length(exp_p_C2_3)),exp_p_C2_3,linspace(1,100,100));
exp_p_C2_4_norm = interp1(linspace(1,100,length(exp_p_C2_4)),exp_p_C2_4,linspace(1,100,100));

figure;
subplot(2,4,1);
plot(exp_i_C_1_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C_1_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie L 1: n = 70');
subplot(2,4,2);
plot(exp_i_C_2_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C_2_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie L 2: n = 63');
subplot(2,4,3);
plot(exp_i_C_3_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C_3_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie L 3: n = 49');
subplot(2,4,4);
plot(exp_i_C_4_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C_4_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie L 4: n = 49');
subplot(2,4,5);
plot(exp_i_C2_1_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C2_1_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie R 1: n = 71');
subplot(2,4,6);
plot(exp_i_C2_2_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C2_2_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie R 2: n = 62');
subplot(2,4,7);
plot(exp_i_C2_3_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C2_3_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie R 3: n = 77');
subplot(2,4,8);
plot(exp_i_C2_4_norm,'Color',[0 1 0]);
hold on;
plot(exp_p_C2_4_norm,'Color',[0 0 1]);
axis tight;
ylim([0 1]);
title('Chewie R 4: n = 73');

per = 20;

figure;
plot(1:per,exp_i_C_1_norm(1:per),'Color',[0 1 0],'LineWidth',4);
hold on;
plot(1:per,exp_p_C_1_norm(1:per),'Color',[0 0 1],'LineWidth',4);
axis tight;
xlabel('percentage of dimensions (%)');
ylabel('explained variance');
title('Chewie 1: until 20%');
ylim([0 1]);

per = 100;

figure;
plot(1:per,exp_i_C_1_norm(1:per),'Color',[0 1 0],'LineWidth',4);
hold on;
plot(1:per,exp_p_C_1_norm(1:per),'Color',[0 0 1],'LineWidth',4);
axis tight;
xlabel('percentage of dimensions (%)');
ylabel('explained variance');
title('Chewie 1: all');
ylim([0 1]);

ed_i_C_1 = calculate_PR(exp_i_C_1);
ed_i_C_2 = calculate_PR(exp_i_C_2);
ed_i_C_3 = calculate_PR(exp_i_C_3);
ed_i_C_4 = calculate_PR(exp_i_C_4);
ed_p_C_1 = calculate_PR(exp_p_C_1);
ed_p_C_2 = calculate_PR(exp_p_C_2);
ed_p_C_3 = calculate_PR(exp_p_C_3);
ed_p_C_4 = calculate_PR(exp_p_C_4);

rd_C_1 = ed_i_C_1/ed_p_C_1;
rd_C_2 = ed_i_C_2/ed_p_C_2;
rd_C_3 = ed_i_C_3/ed_p_C_3;
rd_C_4 = ed_i_C_4/ed_p_C_4;

ed_i_C2_1 = calculate_PR(exp_i_C2_1);
ed_i_C2_2 = calculate_PR(exp_i_C2_2);
ed_i_C2_3 = calculate_PR(exp_i_C2_3);
ed_i_C2_4 = calculate_PR(exp_i_C2_4);
ed_p_C2_1 = calculate_PR(exp_p_C2_1);
ed_p_C2_2 = calculate_PR(exp_p_C2_2);
ed_p_C2_3 = calculate_PR(exp_p_C2_3);
ed_p_C2_4 = calculate_PR(exp_p_C2_4);

rd_C2_1 = ed_i_C2_1/ed_p_C2_1;
rd_C2_2 = ed_i_C2_2/ed_p_C2_2;
rd_C2_3 = ed_i_C2_3/ed_p_C2_3;
rd_C2_4 = ed_i_C2_4/ed_p_C2_4;

rd = [rd_C_1 rd_C_2 rd_C_3 rd_C_4 rd_C2_1 rd_C2_2 rd_C2_3 rd_C2_4];
num_n = [70 63 49 49 71 62 77 73];

figure;
x = categorical({'Chewie L 1','Chewie L 2','Chewie L 3','Chewie L 4','Chewie R 1', ...
    'Chewie R 2','Chewie R 3','Chewie R 4'});
y = [rd_C_1; rd_C_2; rd_C_3; rd_C_4; ...
    rd_C2_1; rd_C2_2; rd_C2_3; rd_C2_4];
b = bar(x,y);
b(1).FaceColor = [0 0 1];
b(1).EdgeColor = [0 0 1];
title('Ratio ISOMAP/PCA');
ylabel('ratio');
ylim([0 1.2]);

R = corrcoef(num_n,rd);
disp(['Pearson correlation coefficient: ' num2str(R(1,2))]);
f = fit(transpose(num_n),transpose(rd),'poly1');
figure;
scatter(num_n,rd,'MarkerEdgeColor',[0 0.5 1], ...
    'MarkerFaceColor',[0 0.5 1],'LineWidth',2);
hold on;
p = plot(f);
set(p,'Color',[0 0 0.8],'LineWidth',3);
title('Relationship between ratio of est. dim. and nº of neurons');
xlabel('nº of neurons');
ylabel('estimated dimensionality');
axis tight;

iter = 10; 
trials = 50;  
dims = 10; 
n_train = 10; 
n_test = 10; 

disp(' -> 10 vs 10...');
disp('     * Chewie L 1');
[exp_i_C_1_M1,exp_p_C_1_M1] = ...
    geodesic(N_C_1_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie L 2');
[exp_i_C_2_M1,exp_p_C_2_M1] = ...
    geodesic(N_C_2_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie L 3');
[exp_i_C_3_M1,exp_p_C_3_M1] = ...
    geodesic(N_C_3_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie L 4');
[exp_i_C_4_M1,exp_p_C_4_M1] = ...
    geodesic(N_C_4_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 1');
[exp_i_C2_1_M1,exp_p_C2_1_M1] = ...
    geodesic(N_C2_1_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 2');
[exp_i_C2_2_M1,exp_p_C2_2_M1] = ...
    geodesic(N_C2_2_M1,dims,n_train,n_test,trials,iter); 
disp('     * Chewie R 3');
[exp_i_C2_3_M1,exp_p_C2_3_M1] = ...
    geodesic(N_C2_3_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 4');
[exp_i_C2_4_M1,exp_p_C2_4_M1] = ...
    geodesic(N_C2_4_M1,dims,n_train,n_test,trials,iter); 

n_train = 20;
n_test = 20;

disp(' -> 20 vs 20...');
disp('     * Chewie L 1');
[exp_i_C_1_M1_2,exp_p_C_1_M1_2] = ...
    geodesic(N_C_1_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie L 2');
[exp_i_C_2_M1_2,exp_p_C_2_M1_2] = ...
    geodesic(N_C_2_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie L 3');
[exp_i_C_3_M1_2,exp_p_C_3_M1_2] = ...
    geodesic(N_C_3_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie L 4');
[exp_i_C_4_M1_2,exp_p_C_4_M1_2] = ...
    geodesic(N_C_4_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 1');
[exp_i_C2_1_M1_2,exp_p_C2_1_M1_2] = ...
    geodesic(N_C2_1_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 2');
[exp_i_C2_2_M1_2,exp_p_C2_2_M1_2] = ...
    geodesic(N_C2_2_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 3');
[exp_i_C2_3_M1_2,exp_p_C2_3_M1_2] = ...
    geodesic(N_C2_3_M1,dims,n_train,n_test,trials,iter);
disp('     * Chewie R 4');
[exp_i_C2_4_M1_2,exp_p_C2_4_M1_2] = ...
    geodesic(N_C2_4_M1,dims,n_train,n_test,trials,iter);

mean_exp_i_C_1 = mean(exp_i_C_1_M1,1);
mean_exp_i_C_2 = mean(exp_i_C_2_M1,1);
mean_exp_i_C_3 = mean(exp_i_C_3_M1,1);
mean_exp_i_C_4 = mean(exp_i_C_4_M1,1);
mean_exp_i_C2_1 = mean(exp_i_C2_1_M1,1);
mean_exp_i_C2_2 = mean(exp_i_C2_2_M1,1);
mean_exp_i_C2_3 = mean(exp_i_C2_3_M1,1);
mean_exp_i_C2_4 = mean(exp_i_C2_4_M1,1);

mean_exp_p_C_1 = mean(exp_p_C_1_M1,1);
mean_exp_p_C_2 = mean(exp_p_C_2_M1,1);
mean_exp_p_C_3 = mean(exp_p_C_3_M1,1);
mean_exp_p_C_4 = mean(exp_p_C_4_M1,1);
mean_exp_p_C2_1 = mean(exp_p_C2_1_M1,1);
mean_exp_p_C2_2 = mean(exp_p_C2_2_M1,1);
mean_exp_p_C2_3 = mean(exp_p_C2_3_M1,1);
mean_exp_p_C2_4 = mean(exp_p_C2_4_M1,1);

std_exp_i_C_1 = sqrt(var(exp_i_C_1_M1,1));
std_exp_i_C_2 = sqrt(var(exp_i_C_2_M1,1));
std_exp_i_C_3 = sqrt(var(exp_i_C_3_M1,1));
std_exp_i_C_4 = sqrt(var(exp_i_C_4_M1,1));
std_exp_i_C2_1 = sqrt(var(exp_i_C2_1_M1,1));
std_exp_i_C2_2 = sqrt(var(exp_i_C2_2_M1,1));
std_exp_i_C2_3 = sqrt(var(exp_i_C2_3_M1,1));
std_exp_i_C2_4 = sqrt(var(exp_i_C2_4_M1,1));

std_exp_p_C_1 = sqrt(var(exp_p_C_1_M1,1));
std_exp_p_C_2 = sqrt(var(exp_p_C_2_M1,1));
std_exp_p_C_3 = sqrt(var(exp_p_C_3_M1,1));
std_exp_p_C_4 = sqrt(var(exp_p_C_4_M1,1));
std_exp_p_C2_1 = sqrt(var(exp_p_C2_1_M1,1));
std_exp_p_C2_2 = sqrt(var(exp_p_C2_2_M1,1));
std_exp_p_C2_3 = sqrt(var(exp_p_C2_3_M1,1));
std_exp_p_C2_4 = sqrt(var(exp_p_C2_4_M1,1));

mean_exp_i_C = mean([mean_exp_i_C_1; mean_exp_i_C_2; ...
    mean_exp_i_C_3; mean_exp_i_C_4],1);
mean_exp_p_C = mean([mean_exp_p_C_1; mean_exp_p_C_2; ...
    mean_exp_p_C_3; mean_exp_p_C_4],1);
mean_exp_i_C2 = mean([mean_exp_i_C2_1; mean_exp_i_C2_2; ...
    mean_exp_i_C2_3; mean_exp_i_C2_4],1);
mean_exp_p_C2 = mean([mean_exp_p_C2_1; mean_exp_p_C2_2; ...
    mean_exp_p_C2_3; mean_exp_p_C2_4],1);
    
std_exp_i_C = sqrt(var([mean_exp_i_C_1; mean_exp_i_C_2; ...
    mean_exp_i_C_3; mean_exp_i_C_4],1));
std_exp_p_C = sqrt(var([mean_exp_p_C_1; mean_exp_p_C_2; ...
    mean_exp_p_C_3; mean_exp_p_C_4],1));
std_exp_i_C2 = sqrt(var([mean_exp_i_C2_1; mean_exp_i_C2_2; ...
    mean_exp_i_C2_3; mean_exp_i_C2_4],1));
std_exp_p_C2 = sqrt(var([mean_exp_p_C2_1; mean_exp_p_C2_2; ...
    mean_exp_p_C2_3; mean_exp_p_C2_4],1));
    
figure;
subplot(2,4,1);
shadedErrorBar(1:dims,mean_exp_i_C_1,std_exp_i_C_1,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C_1,std_exp_p_C_1,'lineprops',{'b'});
axis tight;
ylim([0 0.3]);
subplot(2,4,2);
shadedErrorBar(1:dims,mean_exp_i_C_2,std_exp_i_C_2,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C_2,std_exp_p_C_2,'lineprops',{'b'});
axis tight;
ylim([0 0.3]);
subplot(2,4,3);
shadedErrorBar(1:dims,mean_exp_i_C_3,std_exp_i_C_3,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C_3,std_exp_p_C_3,'lineprops',{'b'});
axis tight;
ylim([0 0.3]);
subplot(2,4,4);
shadedErrorBar(1:dims,mean_exp_i_C_4,std_exp_i_C_4,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C_4,std_exp_p_C_4,'lineprops',{'b'});
axis tight;
ylim([0 0.3]);
subplot(2,4,5);
shadedErrorBar(1:dims,mean_exp_i_C2_1,std_exp_i_C2_1,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C2_1,std_exp_p_C2_1,'lineprops',{'b'});
axis tight;
ylim([0 0.2]);
subplot(2,4,6);
shadedErrorBar(1:dims,mean_exp_i_C2_2,std_exp_i_C2_2,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C2_2,std_exp_p_C2_2,'lineprops',{'b'});
axis tight;
ylim([0 0.2]);
subplot(2,4,7);
shadedErrorBar(1:dims,mean_exp_i_C2_3,std_exp_i_C2_3,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C2_3,std_exp_p_C2_3,'lineprops',{'b'});
axis tight;
ylim([0 0.2]);
subplot(2,4,8);
shadedErrorBar(1:dims,mean_exp_i_C2_4,std_exp_i_C2_4,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C2_4,std_exp_p_C2_4,'lineprops',{'b'});
axis tight;
ylim([0 0.]);
suptitle('Geodesic explained variances (all)');

figure;
subplot(1,2,1);
shadedErrorBar(1:dims,mean_exp_i_C,std_exp_i_C,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C,std_exp_p_C,'lineprops',{'b'});
axis tight;
title('Chewie L');
ylim([0 0.2]);
subplot(1,2,2);
shadedErrorBar(1:dims,mean_exp_i_C2,std_exp_i_C2,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_exp_p_C2,std_exp_p_C2,'lineprops',{'b'});
axis tight;
title('Chewie R');
ylim([0 0.2]);
suptitle('Geodesic explained variances (mean)');

rg_C_1 = mean_exp_i_C_1./mean_exp_p_C_1;
rg_C_2 = mean_exp_i_C_2./mean_exp_p_C_2;
rg_C_3 = mean_exp_i_C_3./mean_exp_p_C_3;
rg_C_4 = mean_exp_i_C_4./mean_exp_p_C_4;
rg_C2_1 = mean_exp_i_C2_1./mean_exp_p_C2_1;
rg_C2_2 = mean_exp_i_C2_2./mean_exp_p_C2_2;
rg_C2_3 = mean_exp_i_C2_3./mean_exp_p_C2_3;
rg_C2_4 = mean_exp_i_C2_4./mean_exp_p_C2_4;

mean_rg_C = mean([rg_C_1; rg_C_2; rg_C_3; rg_C_4],1);
mean_rg_C2 = mean([rg_C2_1; rg_C2_2; rg_C2_3; rg_C2_4],1);
std_rg_C = sqrt(var([rg_C_1; rg_C_2; rg_C_3; rg_C_4],1));
std_rg_C2 = sqrt(var([rg_C2_1; rg_C2_2; rg_C2_3; rg_C2_4],1));

figure;
shadedErrorBar(1:dims,mean_rg_C,std_rg_C,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,mean_rg_C2,std_rg_C2,'lineprops',{'r'});
axis tight;
title('Ratio of geodesic explained variances (ISOMAP / PCA): Chewie L green, Chewie R red');
ylim([0.8 1.8]);

mean_exp_i_C_1_2 = mean(exp_i_C_1_M1_2,1);
mean_exp_p_C_1_2 = mean(exp_p_C_1_M1_2,1);
mean_exp_i_C_2_2 = mean(exp_i_C_2_M1_2,1);
mean_exp_p_C_2_2 = mean(exp_p_C_2_M1_2,1);
mean_exp_i_C_3_2 = mean(exp_i_C_3_M1_2,1);
mean_exp_p_C_3_2 = mean(exp_p_C_3_M1_2,1);
mean_exp_i_C_4_2 = mean(exp_i_C_4_M1_2,1);
mean_exp_p_C_4_2 = mean(exp_p_C_4_M1_2,1);

mean_exp_i_C2_1_2 = mean(exp_i_C2_1_M1_2,1);
mean_exp_p_C2_1_2 = mean(exp_p_C2_1_M1_2,1);
mean_exp_i_C2_2_2 = mean(exp_i_C2_2_M1_2,1);
mean_exp_p_C2_2_2 = mean(exp_p_C2_2_M1_2,1);
mean_exp_i_C2_3_2 = mean(exp_i_C2_3_M1_2,1);
mean_exp_p_C2_3_2 = mean(exp_p_C2_3_M1_2,1);
mean_exp_i_C2_4_2 = mean(exp_i_C2_4_M1_2,1);
mean_exp_p_C2_4_2 = mean(exp_p_C2_4_M1_2,1);

figure;
subplot(2,4,1);
plot(mean_exp_i_C_1,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C_1,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C_1_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C_1_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,2);
plot(mean_exp_i_C_2,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C_2,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C_2_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C_2_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,3);
plot(mean_exp_i_C_3,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C_3,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C_3_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C_3_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,4);
plot(mean_exp_i_C_4,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C_4,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C_4_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C_4_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,5);
plot(mean_exp_i_C2_1,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C2_1,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C2_1_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C2_1_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,6);
plot(mean_exp_i_C2_2,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C2_2,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C2_2_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C2_2_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,7);
plot(mean_exp_i_C2_3,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C2_3,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C2_3_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C2_3_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
subplot(2,4,8);
plot(mean_exp_i_C2_4,'Color',[0 1 0.5],'LineWidth',2);
hold on;
plot(mean_exp_p_C2_4,'Color',[0 0.5 1],'LineWidth',2);
plot(mean_exp_i_C2_4_2,'Color',[0 1 0],'LineWidth',2);
plot(mean_exp_p_C2_4_2,'Color',[0 0 1],'LineWidth',2);
axis tight;
ylim([0 0.4]);
suptitle('Geodesic explained variance (all): different sizes of datasets');

mean_exp_i_C_2 = (mean_exp_i_C_1_2+mean_exp_i_C_2_2+...
    mean_exp_i_C_3_2+mean_exp_i_C_4_2)/4;
mean_exp_p_C_2 = (mean_exp_p_C_1_2+mean_exp_p_C_2_2+...
    mean_exp_p_C_3_2+mean_exp_p_C_4_2)/4;

mean_exp_i_C2_2 = (mean_exp_i_C2_1_2+mean_exp_i_C2_2_2+...
    mean_exp_i_C2_3_2+mean_exp_i_C2_4_2)/4;
mean_exp_p_C2_2 = (mean_exp_p_C2_1_2+mean_exp_p_C2_2_2+...
    mean_exp_p_C2_3_2+mean_exp_p_C2_4_2)/4;

figure;
subplot(1,2,1);
plot(mean_exp_i_C,'Color',[0 1 0],'LineWidth',3);
hold on;
plot(mean_exp_p_C,'Color',[0 0 1],'LineWidth',3);
plot(mean_exp_i_C_2,'Color',[0 0.7 0],'LineWidth',3);
plot(mean_exp_p_C_2,'Color',[0 0 0.7],'LineWidth',3);
axis tight;
ylim([0 0.3]);
subplot(1,2,2);
plot(mean_exp_i_C2,'Color',[0 1 0],'LineWidth',3);
hold on;
plot(mean_exp_p_C2,'Color',[0 0 1],'LineWidth',3);
plot(mean_exp_i_C2_2,'Color',[0 0.7 0],'LineWidth',3);
plot(mean_exp_p_C2_2,'Color',[0 0 0.7],'LineWidth',3);
axis tight;
ylim([0 0.3]);
suptitle('Geodesic explained variance (mean): different sizes of datasets');

num_iter = 50;
dims = 40;
C_C_1_M1 = obtain_correlation_matrix(N_C_1_M1,dims,num_iter,'no');
C_C_2_M1 = obtain_correlation_matrix(N_C_2_M1,dims,num_iter,'no');
C_C_3_M1 = obtain_correlation_matrix(N_C_3_M1,dims,num_iter,'no');
C_C_4_M1 = obtain_correlation_matrix(N_C_4_M1,dims,num_iter,'no');
dims = 40;
C_C2_1_M1 = obtain_correlation_matrix(N_C2_1_M1,dims,num_iter,'no');
C_C2_2_M1 = obtain_correlation_matrix(N_C2_2_M1,dims,num_iter,'no');
C_C2_3_M1 = obtain_correlation_matrix(N_C2_3_M1,dims,num_iter,'no');
C_C2_4_M1 = obtain_correlation_matrix(N_C2_4_M1,dims,num_iter,'no');

C_C_M1 = (C_C_1_M1+C_C_2_M1+C_C_3_M1+C_C_4_M1)/4;
C_C2_M1 = (C_C2_1_M1+C_C2_2_M1+C_C2_3_M1+C_C2_4_M1)/4;

figure;
subplot(1,2,1);
image(abs(C_C_M1)*500);
title('Chewie L');
subplot(1,2,2);
image(abs(C_C2_M1)*500);
title('Chewie R');

figure;
subplot(2,4,1);
image(abs(C_C_1_M1)*500);
subplot(2,4,2);
image(abs(C_C_2_M1)*500);
subplot(2,4,3);
image(abs(C_C_3_M1)*500);
subplot(2,4,4);
image(abs(C_C_4_M1)*500);
subplot(2,4,5);
image(abs(C_C2_1_M1)*500);
subplot(2,4,6);
image(abs(C_C2_2_M1)*500);
subplot(2,4,7);
image(abs(C_C2_1_M1)*500);
subplot(2,4,8);
image(abs(C_C2_2_M1)*500);

vec_C_1_M1 = [];
vec_C_2_M1 = [];
vec_C_3_M1 = [];
vec_C_4_M1 = [];
vec_C2_1_M1 = [];
vec_C2_2_M1 = [];
vec_C2_3_M1 = [];
vec_C2_4_M1 = [];
for i = 1:dims
    for j = 1:dims
        if i ~= j
            vec_C_1_M1 = [vec_C_1_M1 C_C_1_M1(i,j)];
            vec_C_2_M1 = [vec_C_2_M1 C_C_2_M1(i,j)];
            vec_C_3_M1 = [vec_C_3_M1 C_C_3_M1(i,j)];
            vec_C_4_M1 = [vec_C_4_M1 C_C_4_M1(i,j)];
            vec_C2_1_M1 = [vec_C2_1_M1 C_C2_1_M1(i,j)];
            vec_C2_2_M1 = [vec_C2_2_M1 C_C2_2_M1(i,j)];
            vec_C2_3_M1 = [vec_C2_3_M1 C_C2_3_M1(i,j)];
            vec_C2_4_M1 = [vec_C2_4_M1 C_C2_4_M1(i,j)];
        end
    end
end

figure;
subplot(1,2,1);
h1 = histfit(abs(vec_C_1_M1),20,'kernel');
hold on
h2 = histfit(abs(vec_C_2_M1),20,'kernel');
h3 = histfit(abs(vec_C_3_M1),20,'kernel');
h4 = histfit(abs(vec_C_4_M1),20,'kernel');
h1(1).FaceColor = [0 1 0];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 1 0];
h2(1).FaceColor = [0 1 0];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 1 0];
h3(1).FaceColor = [0 1 0];
h3(1).FaceAlpha = 0.4;
h3(1).EdgeAlpha = 0;
h3(2).Color = [0 1 0];
h4(1).FaceColor = [0 1 0];
h4(1).FaceAlpha = 0.4;
h4(1).EdgeAlpha = 0;
h4(2).Color = [0 1 0];
xlabel('correlation');
axis tight;
title('Chewie L');
suptitle('Correlations');
subplot(1,2,2);
h1 = histfit(abs(vec_C2_1_M1),20,'kernel');
hold on
h2 = histfit(abs(vec_C2_2_M1),20,'kernel');
h3 = histfit(abs(vec_C2_3_M1),20,'kernel');
h4 = histfit(abs(vec_C2_4_M1),20,'kernel');
h1(1).FaceColor = [0 0 1];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 0 1];
h2(1).FaceColor = [0 0 1];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 0 1];
h3(1).FaceColor = [0 0 1];
h3(1).FaceAlpha = 0.4;
h3(1).EdgeAlpha = 0;
h3(2).Color = [0 0 1];
h4(1).FaceColor = [0 0 1];
h4(1).FaceAlpha = 0.4;
h4(1).EdgeAlpha = 0;
h4(2).Color = [0 0 1];
xlabel('correlation');
axis tight;
title('Chewie R');
suptitle('Correlations');

vec_C_M1 = [];
vec_C2_M1 = [];

for i = 1:dims
    for j = 1:dims
        if i ~= j
            vec_C_M1 = [vec_C_M1 C_C_M1(i,j)];
            vec_C2_M1 = [vec_C2_M1 C_C2_M1(i,j)];
        end
    end
end

figure;
subplot(1,2,1);
h1 = histfit(abs(vec_C_M1),20,'kernel');
h1(1).FaceColor = [0 1 0];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 1 0];
xlabel('correlation');
axis tight;
title('Chewie L');
subplot(1,2,2);
h1 = histfit(abs(vec_C2_M1),20,'kernel');
h1(1).FaceColor = [0 0 1];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 0 1];
xlabel('correlation');
axis tight;
title('Chewie R');
suptitle('Correlations (mean)');

iter = 5;
trials = 1:50;
dims = 5;

N = A_100_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_100_1 = m_i;
exp_p_100_1 = m_p;

N = A_50_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_50_1 = m_i;
exp_p_50_1 = m_p;

N = A_25_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_25_1 = m_i;
exp_p_25_1 = m_p;

N = A_12_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_12_1 = m_i;
exp_p_12_1 = m_p;

N = A_100_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_100_2 = m_i;
exp_p_100_2 = m_p;

N = A_50_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_50_2 = m_i;
exp_p_50_2 = m_p;

N = A_25_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_25_2 = m_i;
exp_p_25_2 = m_p;

N = A_12_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_12_2 = m_i;
exp_p_12_2 = m_p;

N = A_100_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_100_3 = m_i;
exp_p_100_3 = m_p;

N = A_50_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_50_3 = m_i;
exp_p_50_3 = m_p;

N = A_25_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_25_3 = m_i;
exp_p_25_3 = m_p;

N = A_12_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_12_3 = m_i;
exp_p_12_3 = m_p;

N = A_100_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_100_4 = m_i;
exp_p_100_4 = m_p;

N = A_50_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_50_4 = m_i;
exp_p_50_4 = m_p;

N = A_25_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_25_4 = m_i;
exp_p_25_4 = m_p;

N = A_12_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_12_4 = m_i;
exp_p_12_4 = m_p;

N = A_100_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_100_5 = m_i;
exp_p_100_5 = m_p;

N = A_50_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_50_5 = m_i;
exp_p_50_5 = m_p;

N = A_25_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_25_5 = m_i;
exp_p_25_5 = m_p;

N = A_12_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
exp_i_12_5 = m_i;
exp_p_12_5 = m_p;

exp_i_100 = mean([exp_i_100_1; exp_i_100_2; exp_i_100_3; exp_i_100_4; exp_i_100_5],1);
exp_i_50 = mean([exp_i_50_1; exp_i_50_2; exp_i_50_3; exp_i_50_4; exp_i_50_5],1);
exp_i_25 = mean([exp_i_25_1; exp_i_25_2; exp_i_25_3; exp_i_25_4; exp_i_25_5],1);
exp_i_12 = mean([exp_i_12_1; exp_i_12_2; exp_i_12_3; exp_i_12_4; exp_i_12_5],1);

exp_p_100 = mean([exp_p_100_1; exp_p_100_2; exp_p_100_3; exp_p_100_4; exp_p_100_5],1);
exp_p_50 = mean([exp_p_50_1; exp_p_50_2; exp_p_50_3; exp_p_50_4; exp_p_50_5],1);
exp_p_25 = mean([exp_p_25_1; exp_p_25_2; exp_p_25_3; exp_p_25_4; exp_p_25_5],1);
exp_p_12 = mean([exp_p_12_1; exp_p_12_2; exp_p_12_3; exp_p_12_4; exp_p_12_5],1);

std_exp_i_100 = sqrt(var([exp_i_100_1; exp_i_100_2; exp_i_100_3; exp_i_100_4; exp_i_100_5],1));
std_exp_i_50 = sqrt(var([exp_i_50_1; exp_i_50_2; exp_i_50_3; exp_i_50_4; exp_i_50_5],1));
std_exp_i_25 = sqrt(var([exp_i_25_1; exp_i_25_2; exp_i_25_3; exp_i_25_4; exp_i_25_5],1));
std_exp_i_12 = sqrt(var([exp_i_12_1; exp_i_12_2; exp_i_12_3; exp_i_12_4; exp_i_12_5],1));

std_exp_p_100 = sqrt(var([exp_p_100_1; exp_p_100_2; exp_p_100_3; exp_p_100_4; exp_p_100_5],1));
std_exp_p_50 = sqrt(var([exp_p_50_1; exp_p_50_2; exp_p_50_3; exp_p_50_4; exp_p_50_5],1));
std_exp_p_25 = sqrt(var([exp_p_25_1; exp_p_25_2; exp_p_25_3; exp_p_25_4; exp_p_25_5],1));
std_exp_p_12 = sqrt(var([exp_p_12_1; exp_p_12_2; exp_p_12_3; exp_p_12_4; exp_p_12_5],1));

figure;
subplot(1,4,1);
shadedErrorBar(1:dims,exp_i_100,std_exp_i_100,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,exp_p_100,std_exp_p_100,'lineprops',{'b'});
axis tight;
ylim([0.4 1]);
subplot(1,4,2);
shadedErrorBar(1:dims,exp_i_50,std_exp_i_50,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,exp_p_50,std_exp_p_50,'lineprops',{'b'});
axis tight;
ylim([0.4 1]);
subplot(1,4,3);
shadedErrorBar(1:dims,exp_i_25,std_exp_i_25,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,exp_p_25,std_exp_p_25,'lineprops',{'b'});
axis tight;
ylim([0.4 1]);
subplot(1,4,4);
shadedErrorBar(1:dims,exp_i_12,std_exp_i_12,'lineprops',{'g'});
hold on;
shadedErrorBar(1:dims,exp_p_12,std_exp_p_12,'lineprops',{'b'});
axis tight;
ylim([0.4 1]);

%% GENERAL ANALYSIS: MODEL

load('Net_activity.mat');

trials = 1:50;
sp = 4;
iter = 3;

N = A_100_400_1(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_100_1 = fit(transpose(num),transpose(r),'power2');
f_100_1 = f_100_1(sp:num(end));
N = A_100_400_2(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_100_2 = fit(transpose(num),transpose(r),'power2');
f_100_2 = f_100_2(sp:num(end));
N = A_100_400_3(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_100_3 = fit(transpose(num),transpose(r),'power2');
f_100_3 = f_100_3(sp:num(end));
N = A_100_400_4(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_100_4 = fit(transpose(num),transpose(r),'power2');
f_100_4 = f_100_4(sp:num(end));
N = A_100_400_5(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_100_5 = fit(transpose(num),transpose(r),'power2');
f_100_5 = f_100_5(sp:num(end));

N = A_50_400_1(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_50_1 = fit(transpose(num),transpose(r),'power2');
f_50_1 = f_50_1(sp:num(end));
N = A_50_400_2(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_50_2 = fit(transpose(num),transpose(r),'power2');
f_50_2 = f_50_2(sp:num(end));
N = A_50_400_3(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_50_3 = fit(transpose(num),transpose(r),'power2');
f_50_3 = f_50_3(sp:num(end));
N = A_50_400_4(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_50_4 = fit(transpose(num),transpose(r),'power2');
f_50_4 = f_50_4(sp:num(end));
N = A_50_400_5(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_50_5 = fit(transpose(num),transpose(r),'power2');
f_50_5 = f_50_5(sp:num(end));

N = A_25_400_1(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_25_1 = fit(transpose(num),transpose(r),'power2');
f_25_1 = f_25_1(sp:num(end));
N = A_25_400_2(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_25_2 = fit(transpose(num),transpose(r),'power2');
f_25_2 = f_25_2(sp:num(end));
N = A_25_400_3(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_25_3 = fit(transpose(num),transpose(r),'power2');
f_25_3 = f_25_3(sp:num(end));
N = A_25_400_4(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_25_4 = fit(transpose(num),transpose(r),'power2');
f_25_4 = f_25_4(sp:num(end));
N = A_25_400_5(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_25_5 = fit(transpose(num),transpose(r),'power2');
f_25_5 = f_25_5(sp:num(end));

N = A_12_400_1(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_12_1 = fit(transpose(num),transpose(r),'power2');
f_12_1 = f_12_1(sp:num(end));
N = A_12_400_2(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_12_2 = fit(transpose(num),transpose(r),'power2');
f_12_2 = f_12_2(sp:num(end));
N = A_12_400_3(1:20,:,:);
[r,num] = calculate_ratios(N,iter,trials,sp);
f_12_3 = fit(transpose(num),transpose(r),'power2');
f_12_3 = f_12_3(sp:num(end));
N = A_12_400_4(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_12_4 = fit(transpose(num),transpose(r),'power2');
f_12_4 = f_12_4(sp:num(end));
N = A_12_400_5(1:20,:,:); 
[r,num] = calculate_ratios(N,iter,trials,sp);
f_12_5 = fit(transpose(num),transpose(r),'power2');
f_12_5 = f_12_5(sp:num(end));

mean_f_100 = mean([f_100_1 f_100_2 f_100_3 f_100_4 f_100_5],2);
mean_f_50 = mean([f_50_1 f_50_2 f_50_3 f_50_4 f_50_5],2);
mean_f_25 = mean([f_25_1 f_25_2 f_25_3 f_25_4 f_25_5],2);
mean_f_12 = mean([f_12_1 f_12_2 f_12_3 f_12_4 f_12_5],2);

std_f_100 = sqrt(var([f_100_1 f_100_2 f_100_3 f_100_4 f_100_5],[],2));
std_f_50 = sqrt(var([f_50_1 f_50_2 f_50_3 f_50_4 f_50_5],[],2));
std_f_25 = sqrt(var([f_25_1 f_25_2 f_25_3 f_25_4 f_25_5],[],2));
std_f_12 = sqrt(var([f_12_1 f_12_2 f_12_3 f_12_4 f_12_5],[],2));

figure;
shadedErrorBar(sp:num(end),mean_f_100,std_f_100,'lineprops',{'g'});
hold on;
shadedErrorBar(sp:num(end),mean_f_50,std_f_50,'lineprops',{'c'});
shadedErrorBar(sp:num(end),mean_f_25,std_f_25,'lineprops',{'b'});
shadedErrorBar(sp:num(end),mean_f_12,std_f_12,'lineprops',{'k'});
axis tight;

iter = 10;
trials = 50;
dims = 10;
n_train = 10;
n_test = 10;

disp('# 1 #');
disp(' -> 100 %');
[exp_i_100_1,exp_p_100_1] = geodesic(A_100_400_1,dims,n_train,n_test,trials,iter);
disp(' -> 50 %');
[exp_i_50_1,exp_p_50_1] = geodesic(A_50_400_1,dims,n_train,n_test,trials,iter);
disp(' -> 25 %');
[exp_i_25_1,exp_p_25_1] = geodesic(A_25_400_1,dims,n_train,n_test,trials,iter);
disp(' -> 12.5 %');
[exp_i_12_1,exp_p_12_1] = geodesic(A_12_400_1,dims,n_train,n_test,trials,iter);

disp('# 2 #');
disp(' -> 100 %');
[exp_i_100_2,exp_p_100_2] = geodesic(A_100_400_2,dims,n_train,n_test,trials,iter);
disp(' -> 50 %');
[exp_i_50_2,exp_p_50_2] = geodesic(A_50_400_2,dims,n_train,n_test,trials,iter);
disp(' -> 25 %');
[exp_i_25_2,exp_p_25_2] = geodesic(A_25_400_2,dims,n_train,n_test,trials,iter);
disp(' -> 12.5 %');
[exp_i_12_2,exp_p_12_2] = geodesic(A_12_400_2,dims,n_train,n_test,trials,iter);

disp('# 3 #');
disp(' -> 100 %');
[exp_i_100_3,exp_p_100_3] = geodesic(A_100_400_3,dims,n_train,n_test,trials,iter);
disp(' -> 50 %');
[exp_i_50_3,exp_p_50_3] = geodesic(A_50_400_3,dims,n_train,n_test,trials,iter);
disp(' -> 25 %');
[exp_i_25_3,exp_p_25_3] = geodesic(A_25_400_3,dims,n_train,n_test,trials,iter);
disp(' -> 12.5 %');
[exp_i_12_3,exp_p_12_3] = geodesic(A_12_400_3,dims,n_train,n_test,trials,iter);

disp('# 4 #');
disp(' -> 100 %');
[exp_i_100_4,exp_p_100_4] = geodesic(A_100_400_4,dims,n_train,n_test,trials,iter);
disp(' -> 50 %');
[exp_i_50_4,exp_p_50_4] = geodesic(A_50_400_4,dims,n_train,n_test,trials,iter);
disp(' -> 25 %');
[exp_i_25_4,exp_p_25_4] = geodesic(A_25_400_4,dims,n_train,n_test,trials,iter);
disp(' -> 12.5 %');
[exp_i_12_4,exp_p_12_4] = geodesic(A_12_400_4,dims,n_train,n_test,trials,iter);

disp('# 5 #');
disp(' -> 100 %');
[exp_i_100_5,exp_p_100_5] = geodesic(A_100_400_5,dims,n_train,n_test,trials,iter);
disp(' -> 50 %');
[exp_i_50_5,exp_p_50_5] = geodesic(A_50_400_5,dims,n_train,n_test,trials,iter);
disp(' -> 25 %');
[exp_i_25_5,exp_p_25_5] = geodesic(A_25_400_5,dims,n_train,n_test,trials,iter);
disp(' -> 12.5 %');
[exp_i_12_5,exp_p_12_5] = geodesic(A_12_400_5,dims,n_train,n_test,trials,iter);

mean_exp_i_100 = mean([mean(exp_i_100_1,1); mean(exp_i_100_2,1); ...
    mean(exp_i_100_3,1); mean(exp_i_100_4,1); mean(exp_i_100_5,1)],1);
mean_exp_i_50 = mean([mean(exp_i_50_1,1); mean(exp_i_50_2,1); ...
    mean(exp_i_50_3,1); mean(exp_i_50_4,1); mean(exp_i_50_5,1)],1);
mean_exp_i_25 = mean([mean(exp_i_25_1,1); mean(exp_i_25_2,1); ...
    mean(exp_i_25_3,1); mean(exp_i_25_4,1); mean(exp_i_25_5,1)],1);
mean_exp_i_12 = mean([mean(exp_i_12_1,1); mean(exp_i_12_2,1); ...
    mean(exp_i_12_3,1); mean(exp_i_12_4,1); mean(exp_i_12_5,1)],1);

mean_exp_p_100 = mean([mean(exp_p_100_1,1); mean(exp_p_100_2,1); ...
    mean(exp_p_100_3,1); mean(exp_p_100_4,1); mean(exp_p_100_5,1)],1);
mean_exp_p_50 = mean([mean(exp_p_50_1,1); mean(exp_p_50_2,1); ...
    mean(exp_p_50_3,1); mean(exp_p_50_4,1); mean(exp_p_50_5,1)],1);
mean_exp_p_25 = mean([mean(exp_p_25_1,1); mean(exp_p_25_2,1); ...
    mean(exp_p_25_3,1); mean(exp_p_25_4,1); mean(exp_p_25_5,1)],1);
mean_exp_p_12 = mean([mean(exp_p_12_1,1); mean(exp_p_12_2,1); ...
    mean(exp_p_12_3,1); mean(exp_p_12_4,1); mean(exp_p_12_5,1)],1);

std_exp_i_100 = sqrt(var([mean(exp_i_100_1,1); mean(exp_i_100_2,1); ...
    mean(exp_i_100_3,1); mean(exp_i_100_4,1); mean(exp_i_100_5,1)],1));
std_exp_i_50 = sqrt(var([mean(exp_i_50_1,1); mean(exp_i_50_2,1); ...
    mean(exp_i_50_3,1); mean(exp_i_50_4,1); mean(exp_i_50_5,1)],1));
std_exp_i_25 = sqrt(var([mean(exp_i_25_1,1); mean(exp_i_25_2,1); ...
    mean(exp_i_25_3,1); mean(exp_i_25_4,1); mean(exp_i_25_5,1)],1));
std_exp_i_12 = sqrt(var([mean(exp_i_12_1,1); mean(exp_i_12_2,1); ...
    mean(exp_i_12_3,1); mean(exp_i_12_4,1); mean(exp_i_12_5,1)],1));

std_exp_p_100 = sqrt(var([mean(exp_p_100_1,1); mean(exp_p_100_2,1); ...
    mean(exp_p_100_3,1); mean(exp_p_100_4,1); mean(exp_p_100_5,1)],1));
std_exp_p_50 = sqrt(var([mean(exp_p_50_1,1); mean(exp_p_50_2,1); ...
    mean(exp_p_50_3,1); mean(exp_p_50_4,1); mean(exp_p_50_5,1)],1));
std_exp_p_25 = sqrt(var([mean(exp_p_25_1,1); mean(exp_p_25_2,1); ...
    mean(exp_p_25_3,1); mean(exp_p_25_4,1); mean(exp_p_25_5,1)],1));
std_exp_p_12 = sqrt(var([mean(exp_p_12_1,1); mean(exp_p_12_2,1); ...
    mean(exp_p_12_3,1); mean(exp_p_12_4,1); mean(exp_p_12_5,1)],1));

x = 1:dims;

figure;
subplot(1,3,1);
shadedErrorBar(x,mean_exp_i_100,std_exp_i_100,'lineprops',{'g'});
hold on;
shadedErrorBar(x,mean_exp_i_50,std_exp_i_50,'lineprops',{'c'});
shadedErrorBar(x,mean_exp_i_25,std_exp_i_25,'lineprops',{'b'});
shadedErrorBar(x,mean_exp_i_12,std_exp_i_12,'lineprops',{'r'});
xlabel('dimension');
ylabel('explained variance');
axis tight;
ylim([0.2 0.8]);
h = legend({'100 %','50 %','25 %','12.5 %'},'Location','SouthOutside');
set(h,'Box','off');
title('ISOMAP');
subplot(1,3,2);
shadedErrorBar(x,mean_exp_p_100,std_exp_p_100,'lineprops',{'g'});
hold on;
shadedErrorBar(x,mean_exp_p_50,std_exp_p_50,'lineprops',{'c'});
shadedErrorBar(x,mean_exp_p_25,std_exp_p_25,'lineprops',{'b'});
shadedErrorBar(x,mean_exp_p_12,std_exp_p_12,'lineprops',{'r'});
xlabel('dimension');
ylabel('explained variance');
axis tight;
ylim([0.2 0.8]);
h = legend({'100 %','50 %','25 %','12.5 %'},'Location','SouthOutside');
set(h,'Box','off');
title('PCA');
subplot(1,3,3);
plot(mean_exp_i_100./mean_exp_p_100,'Color',[0 1 0],'LineWidth',2);
hold on;
plot(mean_exp_i_50./mean_exp_p_50,'Color',[0 0.7 1],'LineWidth',2);
plot(mean_exp_i_25./mean_exp_p_25,'Color',[0 0 1],'LineWidth',2);
plot(mean_exp_i_12./mean_exp_p_12,'Color',[1 0 0],'LineWidth',2);
xlabel('dimension');
ylabel('ratio');
axis tight;
ylim([0.4 1.6]);
h = legend({'100 %','50 %','25 %','12.5 %'},'Location','SouthOutside');
set(h,'Box','off');
title('Ratio: Isomap/PCA');

num_iter = 50;
dims = 40;
C_100_1 = obtain_correlation_matrix(A_100_400_1,dims,num_iter,'no');
C_50_1 = obtain_correlation_matrix(A_50_400_1,dims,num_iter,'no');
C_25_1 = obtain_correlation_matrix(A_25_400_1,dims,num_iter,'no');
C_12_1 = obtain_correlation_matrix(A_12_400_1,dims,num_iter,'no');
C_100_2 = obtain_correlation_matrix(A_100_400_2,dims,num_iter,'no');
C_50_2 = obtain_correlation_matrix(A_50_400_2,dims,num_iter,'no');
C_25_2 = obtain_correlation_matrix(A_25_400_2,dims,num_iter,'no');
C_12_2 = obtain_correlation_matrix(A_12_400_2,dims,num_iter,'no');
C_100_3 = obtain_correlation_matrix(A_100_400_3,dims,num_iter,'no');
C_50_3 = obtain_correlation_matrix(A_50_400_3,dims,num_iter,'no');
C_25_3 = obtain_correlation_matrix(A_25_400_3,dims,num_iter,'no');
C_12_3 = obtain_correlation_matrix(A_12_400_3,dims,num_iter,'no');
C_100_4 = obtain_correlation_matrix(A_100_400_4,dims,num_iter,'no');
C_50_4 = obtain_correlation_matrix(A_50_400_4,dims,num_iter,'no');
C_25_4 = obtain_correlation_matrix(A_25_400_4,dims,num_iter,'no');
C_12_4 = obtain_correlation_matrix(A_12_400_4,dims,num_iter,'no');
C_100_5 = obtain_correlation_matrix(A_100_400_5,dims,num_iter,'no');
C_50_5 = obtain_correlation_matrix(A_50_400_5,dims,num_iter,'no');
C_25_5 = obtain_correlation_matrix(A_25_400_5,dims,num_iter,'no');
C_12_5 = obtain_correlation_matrix(A_12_400_5,dims,num_iter,'no');

C_100_mean = (C_100_1+C_100_2+C_100_3+C_100_4+C_100_5)/5;
C_50_mean = (C_50_1+C_50_2+C_50_3+C_50_4+C_50_5)/5;
C_25_mean = (C_25_1+C_25_2+C_25_3+C_25_4+C_25_5)/5;
C_12_mean = (C_12_1+C_12_2+C_12_3+C_12_4+C_12_5)/5;

figure;
subplot(1,4,1);
image(abs(C_12_mean)*500);
title('12.5%');
subplot(1,4,2);
image(abs(C_25_mean)*500);
title('25%');
subplot(1,4,3);
image(abs(C_50_mean)*500);
title('50%');
subplot(1,4,4);
image(abs(C_100_mean)*500);
title('100%');
suptitle('Correlation matrices (means)');

vec_100_1 = [];
vec_50_1 = [];
vec_25_1 = [];
vec_12_1 = [];
vec_100_2 = [];
vec_50_2 = [];
vec_25_2 = [];
vec_12_2 = [];
vec_100_3 = [];
vec_50_3 = [];
vec_25_3 = [];
vec_12_3 = [];
vec_100_4 = [];
vec_50_4 = [];
vec_25_4 = [];
vec_12_4 = [];
vec_100_5 = [];
vec_50_5 = [];
vec_25_5 = [];
vec_12_5 = [];

for i = 1:dims
    for j = 1:dims
        if i ~= j
            vec_100_1 = [vec_100_1 C_100_1(i,j)];
            vec_100_2 = [vec_100_2 C_100_2(i,j)];
            vec_100_3 = [vec_100_3 C_100_3(i,j)];
            vec_100_4 = [vec_100_4 C_100_4(i,j)];
            vec_100_5 = [vec_100_5 C_100_5(i,j)];
            vec_50_1 = [vec_50_1 C_50_1(i,j)];
            vec_50_2 = [vec_50_2 C_50_2(i,j)];
            vec_50_3 = [vec_50_3 C_50_3(i,j)];
            vec_50_4 = [vec_50_4 C_50_4(i,j)];
            vec_50_5 = [vec_50_5 C_50_5(i,j)];
            vec_25_1 = [vec_25_1 C_25_1(i,j)];
            vec_25_2 = [vec_25_2 C_25_2(i,j)];
            vec_25_3 = [vec_25_3 C_25_3(i,j)];
            vec_25_4 = [vec_25_4 C_25_4(i,j)];
            vec_25_5 = [vec_25_5 C_25_5(i,j)];
            vec_12_1 = [vec_12_1 C_12_1(i,j)];
            vec_12_2 = [vec_12_2 C_12_2(i,j)];
            vec_12_3 = [vec_12_3 C_12_3(i,j)];
            vec_12_4 = [vec_12_4 C_12_4(i,j)];
            vec_12_5 = [vec_12_5 C_12_5(i,j)];
        end
    end
end

figure;
h1 = histfit(abs(vec_100_1),20,'kernel');
hold on
h2 = histfit(abs(vec_100_2),20,'kernel');
h3 = histfit(abs(vec_100_4),20,'kernel');
h4 = histfit(abs(vec_100_5),20,'kernel');
h5 = histfit(abs(vec_100_3),20,'kernel');
h6 = histfit(abs(vec_50_1),20,'kernel');
h7 = histfit(abs(vec_50_2),20,'kernel');
h8 = histfit(abs(vec_50_3),20,'kernel');
h9 = histfit(abs(vec_50_4),20,'kernel');
h10 = histfit(abs(vec_50_5),20,'kernel');
h11 = histfit(abs(vec_25_1),20,'kernel');
h12 = histfit(abs(vec_25_2),20,'kernel');
h13 = histfit(abs(vec_25_3),20,'kernel');
h14 = histfit(abs(vec_25_4),20,'kernel');
h15 = histfit(abs(vec_25_5),20,'kernel');
h16 = histfit(abs(vec_12_1),20,'kernel');
h17 = histfit(abs(vec_12_2),20,'kernel');
h18 = histfit(abs(vec_12_3),20,'kernel');
h19 = histfit(abs(vec_12_4),20,'kernel');
h20 = histfit(abs(vec_12_5),20,'kernel'); 
h21 = histfit(abs(vec_C_M1),20,'kernel');
h22 = histfit(abs(vec_C2_M1),20,'kernel');
h1(1).FaceColor = [0 1 0];
h1(1).FaceAlpha = 0.4;
h1(1).EdgeAlpha = 0;
h1(2).Color = [0 1 0];
h2(1).FaceColor = [0 1 0];
h2(1).FaceAlpha = 0.4;
h2(1).EdgeAlpha = 0;
h2(2).Color = [0 1 0];
h3(1).FaceColor = [0 1 0];
h3(1).FaceAlpha = 0.4;
h3(1).EdgeAlpha = 0;
h3(2).Color = [0 1 0];
h4(1).FaceColor = [0 1 0];
h4(1).FaceAlpha = 0.4;
h4(1).EdgeAlpha = 0;
h4(2).Color = [0 1 0];
h5(1).FaceColor = [0 1 0];
h5(1).FaceAlpha = 0.4;
h5(1).EdgeAlpha = 0;
h5(2).Color = [0 1 0];
h6(1).FaceColor = [0 0.66 0.33];
h6(1).FaceAlpha = 0.4;
h6(1).EdgeAlpha = 0;
h6(2).Color = [0 0.66 0.33];
h7(1).FaceColor = [0 0.66 0.33];
h7(1).FaceAlpha = 0.4;
h7(1).EdgeAlpha = 0;
h7(2).Color = [0 0.66 0.33];
h8(1).FaceColor = [0 0.66 0.33];
h8(1).FaceAlpha = 0.4;
h8(1).EdgeAlpha = 0;
h8(2).Color = [0 0.66 0.33];
h9(1).FaceColor = [0 0.66 0.33];
h9(1).FaceAlpha = 0.4;
h9(1).EdgeAlpha = 0;
h9(2).Color = [0 0.66 0.33];
h10(1).FaceColor = [0 0.66 0.33];
h10(1).FaceAlpha = 0.4;
h10(1).EdgeAlpha = 0;
h10(2).Color = [0 0.66 0.33];
h11(1).FaceColor = [0 0.33 0.66];
h11(1).FaceAlpha = 0.4;
h11(1).EdgeAlpha = 0;
h11(2).Color = [0 0.33 0.66];
h12(1).FaceColor = [0 0.33 0.66];
h12(1).FaceAlpha = 0.4;
h12(1).EdgeAlpha = 0;
h12(2).Color = [0 0.33 0.66];
h13(1).FaceColor = [0 0.33 0.66];
h13(1).FaceAlpha = 0.4;
h13(1).EdgeAlpha = 0;
h13(2).Color = [0 0.33 0.66];
h14(1).FaceColor = [0 0.33 0.66];
h14(1).FaceAlpha = 0.4;
h14(1).EdgeAlpha = 0;
h14(2).Color = [0 0.33 0.66];
h15(1).FaceColor = [0 0.33 0.66];
h15(1).FaceAlpha = 0.4;
h15(1).EdgeAlpha = 0;
h15(2).Color = [0 0.33 0.66];
h16(1).FaceColor = [0 0 1];
h16(1).FaceAlpha = 0.4;
h16(1).EdgeAlpha = 0;
h16(2).Color = [0 0 1];
h17(1).FaceColor = [0 0 1];
h17(1).FaceAlpha = 0.4;
h17(1).EdgeAlpha = 0;
h17(2).Color = [0 0 1];
h18(1).FaceColor = [0 0 1];
h18(1).FaceAlpha = 0.4;
h18(1).EdgeAlpha = 0;
h18(2).Color = [0 0 1];
h19(1).FaceColor = [0 0 1];
h19(1).FaceAlpha = 0.4;
h19(1).EdgeAlpha = 0;
h19(2).Color = [0 0 1];
h20(1).FaceColor = [0 0 1];
h20(1).FaceAlpha = 0.4;
h20(1).EdgeAlpha = 0;
h20(2).Color = [0 0 1];
h21(1).FaceColor = [1 0 0];
h21(1).FaceAlpha = 0.4;
h21(1).EdgeAlpha = 0;
h21(2).Color = [1 0 0];
h22(1).FaceColor = [0.7 0 0];
h22(1).FaceAlpha = 0.4;
h22(1).EdgeAlpha = 0;
h22(2).Color = [0.7 0 0];
title('Correlations');

mean_corr = [mean(vec_12_1) mean(vec_25_1) mean(vec_50_1) mean(vec_100_1) ...
    mean(vec_12_2) mean(vec_25_2) mean(vec_50_2) mean(vec_100_2) ...
    mean(vec_12_3) mean(vec_25_3) mean(vec_50_3) mean(vec_100_3) ...
    mean(vec_12_4) mean(vec_25_4) mean(vec_50_4) mean(vec_100_4) ...
    mean(vec_12_5) mean(vec_25_5) mean(vec_50_5) mean(vec_100_5)];

dims = 40;
trials = 1:50;
iter = 5;

N = A_100_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_100_1_i = calculate_PR(m_i);
est_dims_100_1_p = calculate_PR(m_p);
N = A_100_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_100_2_i = calculate_PR(m_i);
est_dims_100_2_p = calculate_PR(m_p);
N = A_100_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_100_3_i = calculate_PR(m_i);
est_dims_100_3_p = calculate_PR(m_p);
N = A_100_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_100_4_i = calculate_PR(m_i);
est_dims_100_4_p = calculate_PR(m_p);
N = A_100_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_100_5_i = calculate_PR(m_i);
est_dims_100_5_p = calculate_PR(m_p);

N = A_50_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_50_1_i = calculate_PR(m_i);
est_dims_50_1_p = calculate_PR(m_p);
N = A_50_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_50_2_i = calculate_PR(m_i);
est_dims_50_2_p = calculate_PR(m_p);
N = A_50_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_50_3_i = calculate_PR(m_i);
est_dims_50_3_p = calculate_PR(m_p);
N = A_50_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_50_4_i = calculate_PR(m_i);
est_dims_50_4_p = calculate_PR(m_p);
N = A_50_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_50_5_i = calculate_PR(m_i);
est_dims_50_5_p = calculate_PR(m_p);

N = A_25_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_25_1_i = calculate_PR(m_i);
est_dims_25_1_p = calculate_PR(m_p);
N = A_25_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_25_2_i = calculate_PR(m_i);
est_dims_25_2_p = calculate_PR(m_p);
N = A_25_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_25_3_i = calculate_PR(m_i);
est_dims_25_3_p = calculate_PR(m_p);
N = A_25_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_25_4_i = calculate_PR(m_i);
est_dims_25_4_p = calculate_PR(m_p);
N = A_25_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_25_5_i = calculate_PR(m_i);
est_dims_25_5_p = calculate_PR(m_p);

N = A_12_400_1; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_12_1_i = calculate_PR(m_i);
est_dims_12_1_p = calculate_PR(m_p);
N = A_12_400_2; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_12_2_i = calculate_PR(m_i);
est_dims_12_2_p = calculate_PR(m_p);
N = A_12_400_3; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_12_3_i = calculate_PR(m_i);
est_dims_12_3_p = calculate_PR(m_p);
N = A_12_400_4; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_12_4_i = calculate_PR(m_i);
est_dims_12_4_p = calculate_PR(m_p);
N = A_12_400_5; 
[m_i,~,m_p,~] = compare_performance(N,iter,dims,trials);
est_dims_12_5_i = calculate_PR(m_i);
est_dims_12_5_p = calculate_PR(m_p);

est_dims_i = [est_dims_12_1_i est_dims_25_1_i est_dims_50_1_i est_dims_100_1_i ...
    est_dims_12_2_i est_dims_25_2_i est_dims_50_2_i est_dims_100_2_i ...
    est_dims_12_3_i est_dims_25_3_i est_dims_50_3_i est_dims_100_3_i ...
    est_dims_12_4_i est_dims_25_4_i est_dims_50_4_i est_dims_100_4_i ...
    est_dims_12_5_i est_dims_25_5_i est_dims_50_5_i est_dims_100_5_i];

est_dims_p = [est_dims_12_1_p est_dims_25_1_p est_dims_50_1_p est_dims_100_1_p ...
    est_dims_12_2_p est_dims_25_2_p est_dims_50_2_p est_dims_100_2_p ...
    est_dims_12_3_p est_dims_25_3_p est_dims_50_3_p est_dims_100_3_p ...
    est_dims_12_4_p est_dims_25_4_p est_dims_50_4_p est_dims_100_4_p ...
    est_dims_12_5_p est_dims_25_5_p est_dims_50_5_p est_dims_100_5_p];

figure;
R = corrcoef(mean_corr,est_dims_i);
disp(['Pearson correlation coefficient (ISOMAP): ' num2str(R(1,2))]);
R = corrcoef(mean_corr,est_dims_p);
disp(['Pearson correlation coefficient (PCA): ' num2str(R(1,2))]);
f_1 = fit(transpose(mean_corr),transpose(est_dims_i),'poly1');
f_2 = fit(transpose(mean_corr),transpose(est_dims_p),'poly1');
scatter(mean_corr,est_dims_i,'MarkerEdgeColor',[0 1 0.5], ...
    'MarkerFaceColor',[0 1 0.5],'LineWidth',2);
hold on;
scatter(mean_corr,est_dims_p,'MarkerEdgeColor',[0 0.5 1], ...
    'MarkerFaceColor',[0 0.5 1],'LineWidth',2);
p_1 = plot(f_1);
set(p_1,'Color',[0 0.8 0],'LineWidth',1);
p_2 = plot(f_2);
set(p_2,'Color',[0 0 0.8],'LineWidth',1);
axis tight;
xlim([min(mean_corr) max(mean_corr)]);
title('Estimated dimensionality (PCA)');
xlabel('mean correlation');
ylabel('estimated dimensionality');
h = legend({'ISOMAP','PCA'},'Location','SouthOutside');
set(h,'Box','off');

ratio_ip = est_dims_i./est_dims_p;

figure;
R = corrcoef(mean_corr,ratio_ip);
disp(['Pearson correlation coefficient (ISOMAP): ' num2str(R(1,2))]);
f = fit(transpose(mean_corr),transpose(ratio_ip),'poly1');
scatter(mean_corr,ratio_ip,'MarkerEdgeColor',[0 1 0.5], ...
    'MarkerFaceColor',[0 1 0.5],'LineWidth',2);
hold on;
p = plot(f);
set(p,'Color',[0 0.8 0],'LineWidth',1);
axis tight;
xlim([min(mean_corr) max(mean_corr)]);
legend hide;

%% BEHAVIOURAL COMPLEXITY ANALYSIS

sp = 5;
iter = 3;

N = N_C2_4_M1;
targets = targets_C2_4_M1;
neurons = size(N,1);

targ_8 = 10; 

N_1 = N(:,find(targets == 1),:);
N_2 = N(:,find(targets == 2),:);
N_3 = N(:,find(targets == 3),:);
N_4 = N(:,find(targets == 4),:);
N_5 = N(:,find(targets == 5),:);
N_6 = N(:,find(targets == 6),:);
N_7 = N(:,find(targets == 7),:);
N_8 = N(:,find(targets == 8),:);

N_1 = N_1(:,1:targ_8,:);
N_2 = N_2(:,1:targ_8,:);
N_3 = N_3(:,1:targ_8,:);
N_4 = N_4(:,1:targ_8,:);
N_5 = N_5(:,1:targ_8,:);
N_6 = N_6(:,1:targ_8,:);
N_7 = N_7(:,1:targ_8,:);
N_8 = N_8(:,1:targ_8,:);

idx = randperm(size(N,2));
N_all_1 = N(:,idx(1:targ_8),:);
idx = randperm(size(N,2));
N_all_2 = N(:,idx(1:targ_8),:);
idx = randperm(size(N,2));
N_all_3 = N(:,idx(1:targ_8),:);
idx = randperm(size(N,2));
N_all_4 = N(:,idx(1:targ_8),:);
idx = randperm(size(N,2));
N_all_5 = N(:,idx(1:targ_8),:);
idx = randperm(size(N,2));
N_all_6 = N(:,idx(1:targ_8),:);

[r_1,num] = calculate_ratios(N_1,iter,1:size(N_1,2),sp);
[r_2,~] = calculate_ratios(N_2,iter,1:size(N_2,2),sp);
[r_3,~] = calculate_ratios(N_3,iter,1:size(N_3,2),sp);
[r_4,~] = calculate_ratios(N_4,iter,1:size(N_4,2),sp);
[r_5,~] = calculate_ratios(N_5,iter,1:size(N_5,2),sp);
[r_6,~] = calculate_ratios(N_6,iter,1:size(N_6,2),sp);
[r_7,~] = calculate_ratios(N_7,iter,1:size(N_7,2),sp);
[r_8,~] = calculate_ratios(N_8,iter,1:size(N_8,2),sp);
[r_all_1,~] = calculate_ratios(N_all_1,iter,1:targ_8,sp);
[r_all_2,~] = calculate_ratios(N_all_2,iter,1:targ_8,sp);
[r_all_3,~] = calculate_ratios(N_all_3,iter,1:targ_8,sp);
[r_all_4,~] = calculate_ratios(N_all_4,iter,1:targ_8,sp);
[r_all_5,~] = calculate_ratios(N_all_5,iter,1:targ_8,sp);
[r_all_6,~] = calculate_ratios(N_all_6,iter,1:targ_8,sp);

f_1 = fit(transpose(num),transpose(r_1),'power2');
f_2 = fit(transpose(num),transpose(r_2),'power2');
f_3 = fit(transpose(num),transpose(r_3),'power2');
f_4 = fit(transpose(num),transpose(r_4),'power2');
f_5 = fit(transpose(num),transpose(r_5),'power2');
f_6 = fit(transpose(num),transpose(r_6),'power2');
f_7 = fit(transpose(num),transpose(r_7),'power2');
f_8 = fit(transpose(num),transpose(r_8),'power2');

f_1_t = zeros(8,neurons-4);
f_1_t(1,:) = f_1(5:neurons);
f_1_t(2,:) = f_2(5:neurons);
f_1_t(3,:) = f_3(5:neurons);
f_1_t(4,:) = f_4(5:neurons);
f_1_t(5,:) = f_5(5:neurons);
f_1_t(6,:) = f_6(5:neurons);
f_1_t(7,:) = f_7(5:neurons);
f_1_t(8,:) = f_8(5:neurons);
mean_f_1_t = mean(f_1_t,1);
std_f_1_t = sqrt(var(f_1_t,1));

f_all_1 = fit(transpose(num),transpose(r_all_1),'power2');
f_all_2 = fit(transpose(num),transpose(r_all_2),'power2');
f_all_3 = fit(transpose(num),transpose(r_all_3),'power2');
f_all_4 = fit(transpose(num),transpose(r_all_4),'power2');
f_all_5 = fit(transpose(num),transpose(r_all_5),'power2');
f_all_6 = fit(transpose(num),transpose(r_all_6),'power2');

f_8_t = zeros(6,neurons-4);
f_8_t(1,:) = f_all_1(5:neurons);
f_8_t(2,:) = f_all_2(5:neurons);
f_8_t(3,:) = f_all_3(5:neurons);
f_8_t(4,:) = f_all_4(5:neurons);
f_8_t(5,:) = f_all_5(5:neurons);
f_8_t(6,:) = f_all_6(5:neurons);
mean_f_8_t = mean(f_8_t,1);
std_f_8_t = sqrt(var(f_8_t,1));

figure;
shadedErrorBar(5:neurons,mean_f_1_t,std_f_1_t,'lineprops',{'b'});
hold on;
shadedErrorBar(5:neurons,mean_f_8_t,std_f_8_t,'lineprops',{'g'});
plot(5:neurons,ones(1,neurons-4),'k--');
axis tight;
title('Ratio ISOMAP / PCA (est. dims.)');
xlabel('number of neurons');
ylabel('ratio');
%ylim([0.3 1.45]);
h = legend({'1 target','8 targets'},'Location','northwest');
set(h,'Box','off');

e_p_1 = estimate_dimensionality(N_1);
e_p_2 = estimate_dimensionality(N_2);
e_p_3 = estimate_dimensionality(N_3);
e_p_4 = estimate_dimensionality(N_4);
e_p_5 = estimate_dimensionality(N_5);
e_p_6 = estimate_dimensionality(N_6);
e_p_7 = estimate_dimensionality(N_7);
e_p_8 = estimate_dimensionality(N_8);
e_p_1_target = mean([e_p_1 e_p_2 e_p_3 e_p_4 e_p_5 e_p_6 e_p_7 e_p_8]);
e_p_all_1 = estimate_dimensionality(N_all_1);
e_p_all_2 = estimate_dimensionality(N_all_2);
e_p_all_3 = estimate_dimensionality(N_all_3);
e_p_all_4 = estimate_dimensionality(N_all_4);
e_p_all_5 = estimate_dimensionality(N_all_5);
e_p_all_6 = estimate_dimensionality(N_all_6);
e_p_8_targets = mean([e_p_all_1 e_p_all_2 e_p_all_3 e_p_all_4 e_p_all_5 e_p_all_6]);

[~,~,exp] = get_isomap(N_1,neurons);
e_i_1 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_2,neurons);
e_i_2 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_3,neurons);
e_i_3 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_4,neurons);
e_i_4 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_5,neurons);
e_i_5 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_6,neurons);
e_i_6 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_7,neurons);
e_i_7 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_8,neurons);
e_i_8 = calculate_PR(exp);
e_i_1_target = mean([e_i_1 e_i_2 e_i_3 e_i_4 e_i_5 e_i_6 e_i_7 e_i_8]);
[~,~,exp] = get_isomap(N_all_1,neurons);
e_i_all_1 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_all_2,neurons);
e_i_all_2 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_all_3,neurons);
e_i_all_3 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_all_4,neurons);
e_i_all_4 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_all_5,neurons);
e_i_all_5 = calculate_PR(exp);
[~,~,exp] = get_isomap(N_all_6,neurons);
e_i_all_6 = calculate_PR(exp);
e_i_8_targets = mean([e_i_all_1 e_i_all_2 e_i_all_3 e_i_all_4 e_i_all_5 e_i_all_6]);

figure;
x = categorical({'1 target','8 targets'});
y = [e_p_1_target e_i_1_target; e_p_8_targets e_i_8_targets];
b = bar(x,y);
b(1).FaceColor = [0 1 0.5];
b(1).EdgeColor = [0 1 0.5];
b(2).FaceColor = [0 0.5 1];
b(2).EdgeColor = [0 0.5 1];
title('Estimated dimensionality');
ylabel('estimated dimensionality');
h = legend({'PCA','ISOMAP'},'Location','northeast');
set(h,'Box','off');

%%

e_p_1_target_C2_4 = e_p_1_target;
e_i_1_target_C2_4 = e_i_1_target;
e_p_8_targets_C2_4 = e_p_8_targets;
e_i_8_targets_C2_4 = e_i_8_targets;
mean_f_1_t_C2_4 = mean_f_1_t;
mean_f_8_t_C2_4 = mean_f_8_t;
std_f_1_t_C2_4 = std_f_1_t;
std_f_8_t_C2_4 = std_f_8_t;
num_C2_4 = num;

%%

figure;
plot(5:70,mean_f_1_t_C_1,'Color',[0 1 0],'LineWidth',3);
hold on;
plot(5:60,mean_f_1_t_C_2(1:end-3),'Color',[0 1 0],'LineWidth',3);
plot(5:45,mean_f_1_t_C_3(1:end-4),'Color',[0 1 0],'LineWidth',3);
plot(5:45,mean_f_1_t_C_4(1:end-4),'Color',[0 1 0],'LineWidth',3);
plot(5:70,mean_f_1_t_C2_1(1:end-1),'Color',[0 1 0],'LineWidth',3);
plot(5:60,mean_f_1_t_C2_2(1:end-2),'Color',[0 1 0],'LineWidth',3);
plot(5:75,mean_f_1_t_C2_3(1:end-2),'Color',[0 1 0],'LineWidth',3);
plot(5:70,mean_f_1_t_C2_4(1:end-3),'Color',[0 1 0],'LineWidth',3);
plot(5:70,mean_f_8_t_C_1,'Color',[0 0 1],'LineWidth',3);
plot(5:60,mean_f_8_t_C_2(1:end-3),'Color',[0 0 1],'LineWidth',3);
plot(5:45,mean_f_8_t_C_3(1:end-4),'Color',[0 0 1],'LineWidth',3);
plot(5:45,mean_f_8_t_C_4(1:end-4),'Color',[0 0 1],'LineWidth',3);
plot(5:70,mean_f_8_t_C2_1(1:end-1),'Color',[0 0 1],'LineWidth',3);
plot(5:60,mean_f_8_t_C2_2(1:end-2),'Color',[0 0 1],'LineWidth',3);
plot(5:75,mean_f_8_t_C2_3(1:end-2),'Color',[0 0 1],'LineWidth',3);
plot(5:70,mean_f_8_t_C2_4(1:end-3),'Color',[0 0 1],'LineWidth',3);
axis tight;
ylim([0.4 1.2]);

r_1_C_1 = e_i_1_target_C_1/e_p_1_target_C_1;
r_1_C_2 = e_i_1_target_C_2/e_p_1_target_C_2;
r_1_C_3 = e_i_1_target_C_3/e_p_1_target_C_3;
r_1_C_4 = e_i_1_target_C_4/e_p_1_target_C_4;
r_1_C2_1 = e_i_1_target_C2_1/e_p_1_target_C2_1;
r_1_C2_2 = e_i_1_target_C2_2/e_p_1_target_C2_2;
r_1_C2_3 = e_i_1_target_C2_3/e_p_1_target_C2_3;
r_1_C2_4 = e_i_1_target_C2_4/e_p_1_target_C2_4;

r_8_C_1 = e_i_8_targets_C_1/e_p_8_targets_C_1;
r_8_C_2 = e_i_8_targets_C_2/e_p_8_targets_C_2;
r_8_C_3 = e_i_8_targets_C_3/e_p_8_targets_C_3;
r_8_C_4 = e_i_8_targets_C_4/e_p_8_targets_C_4;
r_8_C2_1 = e_i_8_targets_C2_1/e_p_8_targets_C2_1;
r_8_C2_2 = e_i_8_targets_C2_2/e_p_8_targets_C2_2;
r_8_C2_3 = e_i_8_targets_C2_3/e_p_8_targets_C2_3;
r_8_C2_4 = e_i_8_targets_C2_4/e_p_8_targets_C2_4;

figure;
x = categorical({'Chewie L 1','Chewie L 2','Chewie L 3','Chewie L 4', ...
    'Chewie R 1','Chewie R 2','Chewie R 3','Chewie R 4'});
y = [r_1_C_1 r_8_C_1; r_1_C_2 r_8_C_2; r_1_C_3 r_8_C_3; r_1_C_4 r_8_C_4; ...
    r_1_C2_1 r_8_C2_1; r_1_C2_2 r_8_C2_2; r_1_C2_3 r_8_C2_3; r_1_C2_4 r_8_C2_4];
b = bar(x,y);
b(1).FaceColor = [0 1 0.5];
b(1).EdgeColor = [0 1 0.5];
b(2).FaceColor = [0 0.5 1];
b(2).EdgeColor = [0 0.5 1];
title('Ratio (ISOMAP/PCA)');
ylabel('ratio');
h = legend({'1 target','8 targets'},'Location','northeast');
set(h,'Box','off');
ylim([0 1.1]);