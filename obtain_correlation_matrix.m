function C_final = obtain_correlation_matrix(X,dims,num_iter,norm)

C_sorted_sum = zeros(dims,dims);

for iter = 1:num_iter
    idx = randperm(size(X,1));
    A = X(idx(1:dims),:,:);
    M = zeros(size(A,1),size(A,2)*size(A,3));
    for i = 1:size(A,2)
        M(:,(size(A,3)*(i-1)+1):(size(A,3)*i)) = squeeze(A(:,i,:));
    end
    C = corrcoef(transpose(M));
    C_sum = zeros(size(C));
    for i = 1:size(A,1)
        [~,idx] = sort(C(i,:));
        C_sum = C_sum + C(idx,idx);
    end
    C_sorted_sum = C_sorted_sum + C_sum/size(A,1);
end

C_final = C_sorted_sum/num_iter;
for i = 1:size(C_final,1)
    C_final(i,i) = nan;
end

if strcmp(norm,'yes')
    C_final = C_final/max(C_final,[],'all');
end