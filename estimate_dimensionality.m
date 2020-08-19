function est_dims = estimate_dimensionality(x)

x_aux = zeros(size(x,2)*size(x,3),size(x,1));

for i = 1:size(x,2)
    for j = 1:size(x,3)
        for k = 1:size(x,1)
            x_aux((i-1)*size(x,3)+j,k) = x(k,i,j);
        end
    end
end

[~,~,eigen] = pca(x_aux);

PR = sum(eigen)^2/sum(eigen.^2); % participation ratio
%disp(eigen);
est_dims = PR; % 1 + round(PR);

end