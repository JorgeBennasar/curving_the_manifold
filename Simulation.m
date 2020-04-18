%% Paths:

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

%% Load and prepare data:

load('Chewie_CO_CS_2016-10-21.mat', 'trial_data');

num_bins = 3; % 30 ms
width_smooth = 0.1;
preparation_bins = [-7, -1]; % -210 ms to -30 ms from idx_movement_on
simulation_bins = [-60, 14]; 
td = binTD(trial_data,num_bins);
td = smoothSignals(td,struct('signals',{'M1_spikes'},'width', ...
    width_smooth));  
td = removeBadTrials(td);
td = removeBadNeurons(td,struct('min_fr',0.1));

% To remove problems with idx_movement_on:

l = length(td);
t_1 = [];
t_2 = [];
t_3 = [];
for i = 1:l
    t_1 = [t_1 td(i).idx_target_on];
    t_2 = [t_2 td(i).idx_movement_on];
    t_3 = [t_3 td(i).idx_movement_on-td(i).idx_target_on];
end

counter = 0;
for i = 1:l
    threshold_x = 0.125*max(abs(td(i-counter).vel(:,1)));
    threshold_y = 0.125*max(abs(td(i-counter).vel(:,2)));
    a = abs(td(i-counter).vel(t_2(i-counter)+1,1) - ...
        td(i-counter).vel(t_2(i-counter),1));
    b = abs(td(i-counter).vel(t_2(i-counter)+1,2) - ...
        td(i-counter).vel(t_2(i-counter),2));
    if (a < threshold_x && b < threshold_y) || (t_3(i-counter) > ...
            -simulation_bins(1)-1) || (t_2(i-counter) <= ...
            -simulation_bins(1))
        td(i-counter) = [];
        t_1(i-counter) = [];
        t_2(i-counter) = [];
        t_3(i-counter) = [];
        counter = counter + 1;
    end
end

disp(['Removed ' num2str(counter) ' trials']);

targets = ones(1,8);
counter = 1;
target_counter = zeros(1,8);

for i = 1:length(td)
    if ismember(td(i).target_direction,targets) == 0
        targets(counter) = td(i).target_direction;
        counter = counter + 1;
    end
    idx = find(targets == td(i).target_direction);
    target_counter(idx) = target_counter(idx) + 1;
    td(i).target = idx;
end

td_p = trimTD(td,{'idx_movement_on',preparation_bins(1)}, ...
    {'idx_movement_on',preparation_bins(2)}); 
td_m = trimTD(td,{'idx_movement_on',simulation_bins(1)}, ...
    {'idx_movement_on',simulation_bins(2)}); 

%% Inputs:

clear param_dim
param_dim.algorithm = 'pca';
param_dim.signals = 'M1_spikes';
param_dim.use_trials = 1:length(td_p);
param_dim.num_dims = 3;
[td_p,~] = dimReduce(td_p,param_dim);

time = simulation_bins(2)-simulation_bins(1)+1;

for i = 1:length(td_p)
    input_p = zeros(param_dim.num_dims, time); % preparatory input
    input_h = zeros(1, time); % hold input
    for j = 1:(simulation_bins(2)-simulation_bins(1)+1)
        if j <= -simulation_bins(1)
            input_h(j) = 0.5;
            if j >= t_1(i) - t_2(i) - simulation_bins(1)
                input_p(:,j) = sum(td_p(i).M1_pca, 1);
            end
        end
    end
    td_m(i).input = [input_p; input_h];
end
    
%% Train: 

percentage_train = 0.7; % w.r.t. the target with less trials

clear param;
param.m_train_per_target = fix(min(target_counter)*percentage_train);
param.mini_batch_size = 8;
param.num_epochs = 80;
param.n_hidden = 40;
param.beta_1 = 0.9;
param.beta_2 = 0.999;
param.epsilon = 1e-8;
param.learning_rate = 0.02;
param.optimization = 'adam';
param.transfer_learning = 'false';
param.transfer_param = 0;
param.r_or_c = 'regression';
param.lambda = 0; % lambda = 0 for no L2 regularization

[net, cost_train, subsets] = runMyLSTM_2(td_m, param);
disp('###########');
disp(['Cost train:' ' ' num2str(cost_train)]);
x_test = subsets.test.X;
y_test = subsets.test.Y;
index_test = subsets.test.index;
disp('###########');

%% Data processing:

time = simulation_bins(2)-simulation_bins(1)+1;
l = size(y_test,2);

A = zeros(param.n_hidden,l,time);
N = zeros(size(td(1).M1_spikes,2),l,time);
y_pred = zeros(size(y_test));

for i = 1:l
    [y_pred_t, A_t] = LSTM_predict(x_test(:,i,:), net, param.r_or_c);
    y_pred(:,i,:) = y_pred_t;
    A(:,i,:) = A_t;
    for j = 1:time
        N(:,i,j) = td_m(index_test(i)).M1_spikes(j,:);
    end
end

% Conversion of activations to all positive values:

A = abs(A);

% Smoothing:

for i = 1:l
    data = smooth_data(transpose(squeeze(A(:,i,:))),td(1).bin_size, ...
        width_smooth);
    for j = 1:size(A,1)
        for k = 1:size(A,3)
            A(j,i,k) = data(k,j);
        end
    end
end

% Normalization of times:

t_3_test = [];
for i = 1:l
    t_3_test = [t_3_test t_3(index_test(i))];
end

L_1 = -max(t_3_test)-simulation_bins(1);
L_2 = min(t_3_test);
time = L_1+L_2+simulation_bins(2)+1;
y_pred_norm = zeros(2,l,time);
y_test_norm = zeros(2,l,time);
A_norm = zeros(param.n_hidden,l,time);
N_norm = zeros(size(td(1).M1_spikes,2),l,time);

for i = 1:l
    L_1_i = -t_3_test(i)-simulation_bins(1);
    L_2_i = t_3_test(i);
    for j = 1:2
        y_pred_norm(j,i,1:L_1) = ...
            resample(squeeze(y_pred(j,i,1:L_1_i)),L_1,L_1_i);
        y_pred_norm(j,i,L_1+1:L_1+L_2) = ...
            resample(squeeze(y_pred(j,i,L_1_i+1:L_1_i+L_2_i)),L_2,L_2_i);
        y_pred_norm(j,i,L_1+L_2+1:time) = ...
            y_pred(j,i,end-simulation_bins(2):end);
        y_test_norm(j,i,1:L_1) = ...
            resample(squeeze(y_test(j,i,1:L_1_i)),L_1,L_1_i);
        y_test_norm(j,i,L_1+1:L_1+L_2) = ...
            resample(squeeze(y_test(j,i,L_1_i+1:L_1_i+L_2_i)),L_2,L_2_i);
        y_test_norm(j,i,L_1+L_2+1:time) = ...
            y_test(j,i,end-simulation_bins(2):end);
    end
    for j = 1:param.n_hidden
        A_norm(j,i,1:L_1) = ...
            resample(squeeze(A(j,i,1:L_1_i)),L_1,L_1_i);
        A_norm(j,i,L_1+1:L_1+L_2) = ...
            resample(squeeze(A(j,i,L_1_i+1:L_1_i+L_2_i)),L_2,L_2_i);
        A_norm(j,i,L_1+L_2+1:time) = A(j,i,end-simulation_bins(2):end);
    end
    for j = 1:size(td(1).M1_spikes,2)
        N_norm(j,i,1:L_1) = resample(squeeze(N(j,i,1:L_1_i)),L_1,L_1_i);
        N_norm(j,i,L_1+1:L_1+L_2) = ...
            resample(squeeze(N(j,i,L_1_i+1:L_1_i+L_2_i)),L_2,L_2_i);
        N_norm(j,i,L_1+L_2+1:time) = N(j,i,end-simulation_bins(2):end);
    end
end

% Elimination of outliers in neurons:

mean_max = mean(max(max(N_norm(:,:,:),[],2),[],3));
std_max = sqrt(var(max(max(N_norm(:,:,:),[],2),[],3)));

counter = 0;
n = size(N,1);
for i = 1:n
    max_neuron = max(max(N_norm(i-counter,:,:),[],2),[],3);
    if max_neuron > mean_max+std_max
        N_norm(i-counter,:,:) = [];
        N(i-counter,:,:) = [];
        counter = counter + 1;
    end
end

%% PSTHs:

PSTH = zeros(size(A,1),8,time);
PSTH_N = zeros(size(N,1),8,time);
idx = [];
for i = 1:8
    counter = 0;
    for j = 1:l
        if td(index_test(j)).target == i
            if length(idx) < i
                idx = [idx j];
            end
            for k = 1:size(A,1)
                PSTH(k,i,:) = PSTH(k,i,:) + A_norm(k,j,:);
            end
            for k = 1:size(N,1)
                PSTH_N(k,i,:) = PSTH_N(k,i,:) + N_norm(k,j,:);
            end
            counter = counter + 1;
        end
    end
    PSTH(:,i,:) = PSTH(:,i,:)/counter;
    PSTH_N(:,i,:) = PSTH_N(:,i,:)/counter;
end

% Normalization of PSTHs and PSTH_Ns:

for i = 1:size(t_3_test)
    PSTH(:,i,:) = PSTH(:,i,:)/max(PSTH(:,i,:),[],'all');
    PSTH_N(:,i,:) = PSTH_N(:,i,:)/max(PSTH_N(:,i,:),[],'all');
end

% Example of PSTHs in target 1 for all RNN units:

figure;
subplot(2,1,1);
for i = 1:size(A,1)
    plot(squeeze(PSTH(i,1,:)),'red');
    hold on;
end
xline(L_1, 'LineWidth',5);
xline(L_1+L_2, 'LineWidth',5);
ylim([-0.5 1.5]);
title('RNN units');

% Example of PSTHs in target 1 for all neurons:

subplot(2,1,2);
for i = 1:1:size(N,1)
    plot(squeeze(PSTH_N(i,1,:)),'blue');
    hold on;
end
xline(L_1, 'LineWidth',5);
xline(L_1+L_2, 'LineWidth',5);
ylim([-0.5 1.5]);
title('Neurons');
suptitle('Example of PSTHs in target 1');


%% dPSTHs:

% Mean of absolute value of dPSTHs (rate of change) for RNN units and neurons:

dPSTH = zeros(size(PSTH(:,:,end-9)));
dPSTH_N = zeros(size(PSTH_N(:,:,end-9)));
for i = 10:time
    dPSTH(:,:,i-9) = PSTH(:,:,i) - PSTH(:,:,i-1);
    dPSTH_N(:,:,i-9) = PSTH_N(:,:,i) - PSTH_N(:,:,i-1);
end

figure;
plot(squeeze(mean(mean(abs(dPSTH),1),2)),'red','LineWidth',2);
hold on;
plot(squeeze(mean(mean(abs(dPSTH_N),1),2)),'blue','LineWidth',2);
hold on;
xline(L_1+L_2-9, 'LineWidth',5);
title('Mean of absolute value of dPSTHs for RNN units and neurons');

%% Activations:

% Activations of RNN unit 1 in all test trials with targets 1, 2 and 3:

unit = 1;

figure;
aux = zeros(3,time);
counter = zeros(1,3);
color = {'blue' 'red' 'green'};
for i = 1:l
    for j = 1:3
        if td(index_test(i)).target == j
            plot(squeeze(A_norm(unit,i,:)),color{(j)});
            hold on;
            aux(j,:) = aux(j,:) + transpose(squeeze(A_norm(unit,i,:)));
            counter(j) = counter(j) + 1;
        end
    end
end
aux(1,:) = aux(1,:)/counter(1);
aux(2,:) = aux(2,:)/counter(2);
aux(3,:) = aux(3,:)/counter(3);
plot(aux(1,:),'Color',[0 0 0.7],'LineWidth',3);
hold on;
plot(aux(2,:),'Color',[0.7 0 0],'LineWidth',3);
hold on;
plot(aux(3,:),'Color',[0 0.7 0],'LineWidth',3);
hold on;
xline(L_1, 'LineWidth',5);
hold on;
xline(L_1+L_2, 'LineWidth',5);
title('Activations of unit 1 in all test trials with targets 1, 2 and 3');

%% Velocity:

time_plot = linspace(1,time*l,time*l);

y_test_plot = zeros(2,time*l);
y_pred_plot = zeros(2,time*l);

for i = 1:time
    for j = 1:l
        y_test_plot(1,(j-1)*time+i) = y_test_norm(1,j,i);
        y_pred_plot(1,(j-1)*time+i) = y_pred_norm(1,j,i);
        y_test_plot(2,(j-1)*time+i) = y_test_norm(2,j,i);
        y_pred_plot(2,(j-1)*time+i) = y_pred_norm(2,j,i);
    end
end

vaf_x = compute_vaf(transpose(y_test_plot(1,:)), ...
    transpose(y_pred_plot(1,:)));
vaf_y = compute_vaf(transpose(y_test_plot(2,:)), ...
    transpose(y_pred_plot(2,:)));

% Velocity (predicted and real):

figure;
ax(1) = subplot(2,1,1); hold all;
plot(y_test_plot(1,:),'LineWidth',2);
plot(y_pred_plot(1,:),'LineWidth',2);
title(['VAF = ' num2str(vaf_x,3)]);
axis tight;
ax(2) = subplot(2,1,2); hold all;
plot(y_test_plot(2,:),'LineWidth',2);
plot(y_pred_plot(2,:),'LineWidth',2);
title(['VAF = ' num2str(vaf_y,3)]);
axis tight;
h = legend({'Actual','Predicted'},'Location','SouthOutside');
set(h,'Box','off');
linkaxes(ax,'x');
suptitle('Velocity');

%{

%% Rotational dynamics:

dims_pca = 2;
A_pca = get_pca(A_norm,dims_pca);

% Rotational dynamics for all targets (RNN units):

figure;
axis('tight');
set(gca,'Box','off','TickDir','out','FontSize',14);
for i = 1:8
    for j = 1:l
        if td(index_test(j)).target == i
            subplot(3,3,i)
            plot(A_pca(1,j,L_1),A_pca(2,j,L_1),'ro');
            hold on;
            plot(A_pca(1,j,L_1+L_2),A_pca(2,j,L_1+L_2),'bo');
            hold on;
            plot(A_pca(1,j,end),A_pca(2,j,end),'go');
            hold on;
            plot(squeeze(A_pca(1,j,:)),squeeze(A_pca(2,j,:)),'Color','k');
        end
    end
end
suptitle('Rotational dynamics for all targets (RNN units)');

%}

%% Preparatory activity:

dims_pca = 3;
A_pca = get_pca(A_norm,dims_pca);
% N_pca = get_pca(N_norm,dims_pca);

A_pca_prep = A_pca(:,:,L_1+L_2-7:L_1+L_2);
A_pca_sum = sum(A_pca_prep,3);
% N_pca_prep = N_pca(:,:,L_1+L_2-7:L_1+L_2);
% N_pca_sum = sum(N_pca_prep,3);

color = {[0,0,1] [0,0.5,0] [0.3010,0.7450,0.9330] ...
    [0.25,0.25,0.25] [0.9290,0.6940,0.1250] [0.75,0,0.75] ...
    [0.6350,0.0780,0.1840] [0.8500,0.3250,0.0980]};

% First three PC of preparatory activity per trial of RNN units:

figure;
subplot(1,2,1)
for i = 1:l
    plot3(A_pca_sum(1,i),A_pca_sum(2,i),A_pca_sum(3,i),'.', ...
        'Color',color{(td(index_test(i)).target)},'MarkerSize',20);
    hold on;
end
title('First three PC of preparatory activity per trial of RNN units');

% First three PC of preparatory activity per trial of neurons: 

subplot(1,2,2)
for i = 1:l
    plot3(td_m(index_test(i)).input(1,t_1(index_test(i))- ...
        t_2(index_test(i))-simulation_bins(1)), ...
        td_m(index_test(i)).input(2,t_1(index_test(i))- ...
        t_2(index_test(i))-simulation_bins(1)), ...
        td_m(index_test(i)).input(3,t_1(index_test(i))- ...
        t_2(index_test(i))-simulation_bins(1)),'.', ...
        'Color',color{(td(index_test(i)).target)},'MarkerSize',20);
    hold on;
end
title('First three PC of preparatory activity per trial of neurons');

% Naive Bayes:

X = zeros(l,3);
Y = zeros(l,1);
for i = 1:l
    X(i,:) = transpose(A_pca_sum(:,i)); 
    Y(i) = td(index_test(i)).target;
end

cross_val = 10;

NB_model = fitcnb(X,Y);
NB_cvmodel = crossval(NB_model,'KFold',cross_val);
L = kfoldLoss(NB_cvmodel,'lossfun','classiferror');
disp(['Naive Bayes classification error: ' num2str(L*100) ' %']);

% KNN:

k = 3;
KNN_model = fitcknn(X,Y,'NumNeighbors',k,'Standardize',1);
KNN_cvmodel = crossval(KNN_model,'KFold',cross_val);
L = kfoldLoss(KNN_cvmodel,'lossfun','classiferror');
disp(['KNN classification error: ' num2str(L*100) ' %']);

%% Correlation between PSTHs and PSTH_Ns:

% For target 1:

r = zeros(size(PSTH,1),size(PSTH_N,1));
for i = 1:size(PSTH,1)
    for j = 1:size(PSTH_N,1)
        C = corrcoef(PSTH(i,1,:),PSTH_N(j,1,:));
        r(i,j) = C(1,2);
    end
end

v_1 = zeros(1,size(PSTH,1));
v_1_idx = zeros(1,size(PSTH,1));
for i = 1:size(PSTH,1)
    v_1(i) = max(r(i,:));
    v_1_idx(i) = find(v_1(i) == r(i,:));
end

k = 20;

max_k = maxk(v_1,k);
idx_max_k = zeros(1,k);
for i = 1:k
    idx_max_k(i) = find(max_k(i) == v_1);
end

figure;
counter = 1;
for i = idx_max_k
    subplot(fix(sqrt(k)+1),fix(sqrt(k)+1),counter);
    plot(squeeze(PSTH(i,1,:)),'red');
    hold on;
    plot(squeeze(PSTH_N(v_1_idx(i),1,:)),'blue');
    hold on;
    counter = counter + 1;
end
suptitle('Highly correlated pairs of neurons and RNN units (PSTHs)');

%% Canonical correlation analysis (activations and neural activity):

AA = zeros(size(A,1),l*time);
NN = zeros(size(N,1),l*time);
for i = 1:l
    AA(:,(time*(i-1)+1):(time*i)) = squeeze(A_norm(:,i,:));
    NN(:,(time*(i-1)+1):(time*i)) = squeeze(N_norm(:,i,:));
end

dims_pca = 10;
AA_pca = transpose(get_pca(AA,dims_pca));
NN_pca = transpose(get_pca(NN,dims_pca));
[~,~,r_1,U,V] = canoncorr(AA_pca,NN_pca);

A_CV_mean = zeros(dims_pca,8,time);
N_CV_mean = zeros(dims_pca,8,time);
counter = zeros(1,8);
for i = 1:8
    A_s = zeros(dims_pca,time);
    N_s = zeros(dims_pca,time);
    for j = 1:l
        if td(index_test(j)).target == i
            for k = 1:dims_pca
                A_s(k,:) = A_s(k,:) + ...
                    transpose(U((time*(j-1)+1):(time*j),k));
                N_s(k,:) = N_s(k,:) + ...
                    transpose(V((time*(j-1)+1):(time*j),k));
            end
            counter(i) = counter(i) + 1;
        end
    end
    A_CV_mean(:,i,:) = A_s/counter(i);
    N_CV_mean(:,i,:) = N_s/counter(i);
end

dims_plot = [1 2 5 9 10];

figure;
flag = 1;
for target = 1:8
    counter = 1;
    for i = dims_plot
        subplot(length(dims_plot),2,2*counter-1);
        plot(transpose(squeeze(N_CV_mean(i,target,L_1+L_2-5:end))),'red');
        hold on;
        xlim([-6 simulation_bins(2)+7]);
        ylim([-4 4]);
        if flag == 1
            txt = ['r = ' num2str(round(r_1(i),2))];
            text(-5,3,txt);
        end
        title(['Neural data: CCA projection ' num2str(i)]);
        subplot(length(dims_plot),2,2*counter);
        plot(transpose(squeeze(A_CV_mean(i,target,L_1+L_2-5:end))),'blue');
        hold on;
        xlim([-6 simulation_bins(2)+7]);
        ylim([-4 4]);
        title(['RNN activity: CCA projection ' num2str(i)]);
        counter = counter + 1;
    end
    flag = 0;
end
suptitle('Canonical correlation analysis');

figure;
plot(r_1,'Color',[0 1 0],'LineWidth',3);
title('r value in CCA');
             
%% Canonical correlation analysis (PSTHs and PSTH_Ns):

PSTH_CCA = zeros(size(PSTH,1),8*time);
PSTH_N_CCA = zeros(size(PSTH_N,1),8*time);
for i = 1:8
    PSTH_CCA(:,(time*(i-1)+1):(time*i)) = squeeze(PSTH(:,i,:));
    PSTH_N_CCA(:,(time*(i-1)+1):(time*i)) = squeeze(PSTH_N(:,i,:));
end

dims_pca = 10;
PSTH_pca = transpose(get_pca(PSTH_CCA,dims_pca));
PSTH_N_pca = transpose(get_pca(PSTH_N_CCA,dims_pca));
[~,~,r_2,U,V] = canoncorr(PSTH_pca,PSTH_N_pca);

PSTH_CV_mean = zeros(dims_pca,8,time);
PSTH_N_CV_mean = zeros(dims_pca,8,time);
for i = 1:8
    for k = 1:dims_pca
        PSTH_CV_mean(k,i,:) = transpose(U((time*(i-1)+1):(time*i),k));
        PSTH_N_CV_mean(k,i,:) = transpose(V((time*(i-1)+1):(time*i),k));
    end
end

dims_plot = [1 2 5 9 10];

figure;
flag = 1;
for target = 1:8
    counter = 1;
    for i = dims_plot
        subplot(length(dims_plot),2,2*counter-1);
        plot(transpose(squeeze( ...
            PSTH_N_CV_mean(i,target,L_1+L_2-5:end))),'red');
        hold on;
        xlim([-6 simulation_bins(2)+7]);
        ylim([-4 4]);
        if flag == 1
            txt = ['r = ' num2str(round(r_2(i),2))];
            text(-5,3,txt);
        end
        title(['Neural data: CCA projection ' num2str(i)]);
        subplot(length(dims_plot),2,2*counter);
        plot(transpose(squeeze(PSTH_CV_mean(i,target,L_1+L_2-5:end))), ...
            'blue');
        hold on;
        xlim([-6 simulation_bins(2)+7]);
        ylim([-4 4]);
        title(['RNN activity: CCA projection ' num2str(i)]);
        counter = counter + 1;
    end
    flag = 0;
end
suptitle('Canonical correlation analysis');

figure;
plot(r_2,'Color',[1 0 0],'LineWidth',3);
title('r value in CCA');

%% r-value comparison between CCAs:

figure;
plot(r_1,'Color',[0 1 0],'LineWidth',3);
hold on;
plot(r_2,'Color',[1 0 0],'LineWidth',3);
title('r values in CCA');
h = legend({'Activations','PSTHs'},'Location','SouthOutside');
set(h,'Box','off');
