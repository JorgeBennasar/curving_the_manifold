function [x_pca] = get_pca(x,dims)

x_aux = zeros(size(x,2)*size(x,3),size(x,1));

for i = 1:size(x,2)
    for j = 1:size(x,3)
        for k = 1:size(x,1)
            x_aux((i-1)*size(x,3)+j,k) = x(k,i,j);
        end
    end
end

coeff = pca(x_aux);

x_pca = zeros(size(x));

for i = 1:size(x,1)
    for j = 1:size(x,2)
        for k = 1:size(x,3)
            x_pca(i,j,k) = sum(transpose(x(:,j,k))*coeff(:,i));
        end
    end
end

x_pca = x_pca(1:dims,:,:);

end