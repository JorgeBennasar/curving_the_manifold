function [x_pca,exp_var_pca,exp_var_pca_2,coeff] = get_pca(x,dims)

x_aux = zeros(size(x,2)*size(x,3),size(x,1));

for i = 1:size(x,2)
    for j = 1:size(x,3)
        for k = 1:size(x,1)
            x_aux((i-1)*size(x,3)+j,k) = x(k,i,j);
        end
    end
end

D = L2_distance(x_aux',x_aux');

[coeff,~,eigen] = pca(x_aux);
%disp(eigen);

x_pca = zeros(size(x));

for i = 1:size(x,1)
    for j = 1:size(x,2)
        for k = 1:size(x,3)
            x_pca(i,j,k) = sum(transpose(x(:,j,k))*coeff(:,i));
        end
    end
end

x_pca = x_pca(1:dims,:,:);

exp_var_pca = zeros(1,size(x,1));
for i = 1:size(x,1)
    exp_var_pca(i) = sum(eigen(1:i))/sum(eigen);
end

N = size(D,1);
%{
opt.disp = 0; 
[vec, val] = eigs(-.5*(D.^2 - sum(D.^2)'*ones(1,N)/N - ...
    ones(N,1)*sum(D.^2)/N + sum(sum(D.^2))/(N^2)), dims, 'LR', opt); 
h = real(diag(val)); 
[foo,sorth] = sort(h);  sorth = sorth(end:-1:1); 
val = real(diag(val(sorth,sorth))); 
vec = vec(:,sorth); 
%}
D = reshape(D,N^2,1);
x_pca_aux = A_to_AA(x_pca);

R = zeros(1,dims);
for di = 1:dims
     Y = x_pca_aux(1:di,:); % real(vec(:,1:di).*(ones(N,1)*sqrt(val(1:di))'))';
     r2 = 1-corrcoef(reshape(real(L2_distance(Y,Y)),N^2,1),D).^2; 
     R(di) = r2(2,1); 
     %disp(['  PCA with dimensionality ', num2str(di), '  --> residual variance = ', num2str(R(di))]); 
end

exp_var_pca_2 = zeros(1,dims);
for i = 1:dims
    exp_var_pca_2(i) = 1 - R(i);
end

end