function [mean_e_i,std_e_i,mean_e_p,std_e_p] = ...
    compare_performance(N,iter,dims,trials)

e_i = zeros(iter,dims);
e_p = zeros(iter,dims);

for it = 1:iter
    idx_t = randperm(size(N,2));
    idx_d = randperm(size(N,1));
    [~,~,e_i(it,:)] = get_isomap(N(idx_d(1:dims),idx_t(trials),:),dims);
    [~,e_p(it,:),~,~] = get_pca(N(idx_d(1:dims),idx_t(trials),:),dims);
end

x = 1:dims;

mean_e_i = mean(e_i,1);
std_e_i = sqrt(var(e_i,1));
mean_e_p = mean(e_p,1);
std_e_p = sqrt(var(e_p,1));
%{
figure;
if iter > 1
    shadedErrorBar(x,mean_e_i,std_e_i,'lineprops',{'g'});
    hold on;
    shadedErrorBar(x,mean_e_p,std_e_p,'lineprops',{'b'});
else
    plot(mean_e_i,'Color',[0 1 0],'LineWidth',3);
    hold on;
    plot(mean_e_p,'Color',[0 0 1],'LineWidth',3);
end
axis tight;
xlabel('dimension');
ylabel('explained variance');
title('Explained variance');
h = legend({'ISOMAP','PCA'},'Location','southeast');
set(h,'Box','off');
%}
end