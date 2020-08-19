function [x_isomap,exp_var_isomap,exp_var_eigen] = get_isomap(x,num_dims)

iso_n = 20;
iso_function = 'k';    
iso_opts = struct('dims',1:num_dims,'comp',1,'display',false, ...
    'overlay',true,'verbose',true);

x_concat = zeros(size(x,2)*size(x,3),size(x,1));
for i = 1:size(x,1)
    for j = 1:size(x,2)
        for k = 1:size(x,3)
            x_concat((j-1)*size(x,3)+k,i) = x(i,j,k);
        end
    end
end

D = L2_distance(x_concat',x_concat');

[Y,R,~,N_1,N_2,val] = isomap(D,iso_function,iso_n,iso_opts);

scores = Y.coords{end}';

if N_1 == N_2
    x_isomap = zeros(num_dims,size(x,2),size(x,3));
    for i = 1:size(x,2)
        for j = 1:size(x,3)
            for k = 1:num_dims
                x_isomap(k,i,j) = scores((i-1)*size(x,3)+j,k);
            end
        end
    end
else
    disp('Error: cannot compute matrix, only explained variance given');
    x_isomap = 'Not computed';
end

exp_var_isomap = zeros(1,num_dims);
for i = 1:num_dims
    exp_var_isomap(i) = 1 - R(i);
end

exp_var_eigen = zeros(1,num_dims);
for i = 1:num_dims
    exp_var_eigen(i) = sum(val(1:i))/sum(val);
end

%{
D = D(Y.index, Y.index);
D_new = reshape(D,size(D,1)^2,1);
x_isomap_aux = A_to_AA(x_isomap);
R_2 = zeros(1,num_dims);
for di = 1:num_dims
    Y = x_isomap_aux(1:di,:);
    r2 = 1-corrcoef(reshape(real(L2_distance(Y,Y)),size(D,1)^2,1),D_new).^2; 
    R_2(di) = r2(2,1); 
    disp(['  Isomap with dimensionality ', num2str(di), '  --> residual variance = ', num2str(R_2(di))]); 
end

exp_var_isomap_2 = zeros(1,num_dims);
for i = 1:num_dims
    exp_var_isomap_2(i) = 1 - R_2(i);
end
%}
end