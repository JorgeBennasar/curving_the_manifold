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

load('center_out_data.mat', 'trial_data');

td = smoothSignals(trial_data,struct('signals',{'M1_spikes'},'width',0.1));  
td = removeBadTrials(td);
td = trimTD(td, {'idx_target_on',-5},'idx_trial_end'); 
td = removeBadNeurons(td,struct('min_fr',0.5));

clear param;
param.dim_reduce_method = 'isomap';
param.dims = 5;
param.percentage_train = 0.9;
param.mini_batch_size = 32;
param.num_epochs = 50;
param.n_hidden = 40;
param.beta_1 = 0.9;
param.beta_2 = 0.999;
param.epsilon = 1e-8;
param.learning_rate = 0.05;
param.optimization = 'adam';
param.transfer_learning = 'false';
param.transfer_param = 0;
param.r_or_c = 'regression';

if strcmp(param.dim_reduce_method,'isomap')
    load('td_isomap.mat');
    for i = 1:length(td)
        td(i).M1_isomap = td_new(i).M1_isomap(:,1:param.dims);
    end
end

%% Train:

[net, cost_train, subsets, min_length] = runMyLSTM(td, param);
disp('###########');
disp(['Cost train:' ' ' num2str(cost_train)]);
x_test = subsets.test.X;
y_test = subsets.test.Y;
[y_pred, A, cost_test] = LSTM_predict(x_test, y_test, ...
    net, param.r_or_c);
disp(['Cost test:' ' ' num2str(cost_test)]);
disp('###########');

%% Plots:

l = m_test;

y_test_l = zeros(2,min_length*l);
y_pred_l = zeros(2,min_length*l);

for i = 1:min_length
    for j = 1:l
        y_test_l(1,(j-1)*min_length+i) = y_test(1,j,i);
        y_pred_l(1,(j-1)*min_length+i) = y_pred(1,j,i);
        y_test_l(2,(j-1)*min_length+i) = y_test(2,j,i);
        y_pred_l(2,(j-1)*min_length+i) = y_pred(2,j,i);
    end
end

vaf_x = compute_vaf(transpose(y_test_l(1,:)),transpose(y_pred_l(1,:)));
vaf_y = compute_vaf(transpose(y_test_l(2,:)),transpose(y_pred_l(2,:)));

figure;
ax(1) = subplot(2,1,1); hold all;
plot(y_test_l(1,:),'LineWidth',2);
plot(y_pred_l(1,:),'LineWidth',2);
title(['VAF = ' num2str(vaf_x,3)]);
axis tight;
ax(2) = subplot(2,1,2); hold all;
plot(y_test_l(2,:),'LineWidth',2);
plot(y_pred_l(2,:),'LineWidth',2);
title(['VAF = ' num2str(vaf_y,3)]);
axis tight;
h = legend({'Actual','Predicted'},'Location','SouthOutside');
set(h,'Box','off');
linkaxes(ax,'x');
suptitle('Position (X and Y)');