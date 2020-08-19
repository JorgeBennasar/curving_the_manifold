function [exp_i,exp_p] = geodesic(A,dims,n_train,n_test,trials,iter)

iso_n = 20;
iso_function = 'k';    
iso_opts = struct('dims',1:dims,'comp',1,'display',false, ...
    'overlay',true,'verbose',true);

exp_i = zeros(iter,dims);
exp_p = zeros(iter,dims);

for it = 1:iter
    idx = randperm(size(A,1));
    idx_t = randperm(size(A,2));
    x = A(idx(1:n_train),idx_t(1:trials),:);
    y = A(idx(n_train+1:n_train+n_test),idx_t(1:trials),:);
    
    %Y_pred = A_to_AA(y);
    
    y_aux = A_to_AA(y);
    D_y = L2_distance(y_aux,y_aux);

    [y_pca,~,~,~] = get_pca(y,dims);
    y_pca = A_to_AA(y_pca);
    
    [Y,~,~,N_1,N_2] = isomap(D_y,iso_function,iso_n,iso_opts);
    scores = Y.coords{end}';
    if N_1 == N_2
        y_isomap = zeros(dims,size(y,2),size(y,3));
        for i = 1:size(y,2)
            for j = 1:size(y,3)
                for k = 1:dims
                    y_isomap(k,i,j) = scores((i-1)*size(y,3)+j,k);
                end
            end
        end
    else
        disp('Error: cannot compute matrix, only explained variance given');
        y_isomap = 'Not computed';
        return;
    end
    y_isomap = A_to_AA(y_isomap);
        
    x_aux = A_to_AA(x);
    D_x = L2_distance(x_aux,x_aux);
    N = size(D_x,1);
    
    % Isomap:

    [Y,~,~,N_1,N_2] = isomap(D_x,iso_function,iso_n,iso_opts);
    scores = Y.coords{end}';
    if N_1 == N_2
        x_isomap = zeros(dims,size(x,2),size(x,3));
        for i = 1:size(x,2)
            for j = 1:size(x,3)
                for k = 1:dims
                    x_isomap(k,i,j) = scores((i-1)*size(x,3)+j,k);
                end
            end
        end
    else
        disp('Error: cannot compute matrix, only explained variance given');
        x_isomap = 'Not computed';
        return;
    end
    x_isomap_aux = A_to_AA(x_isomap);
    for di = 1:dims
        Y = x_isomap_aux(1:di,:);
        Y_pred = y_isomap(1:dims,:); %Y_pred = y_isomap(1:di,:);
        r2 = corrcoef(reshape(real(L2_distance(Y,Y)),N^2,1), ...
            reshape(real(L2_distance(Y_pred,Y_pred)),N^2,1)).^2; 
        exp_i(it,di) = r2(1,2);
    end

    % PCA:

    [coeff,~,~] = pca(x_aux');
    x_pca = zeros(size(x));
    for i = 1:size(x,1)
        for j = 1:size(x,2)
            for k = 1:size(x,3)
                x_pca(i,j,k) = sum(transpose(x(:,j,k))*coeff(:,i));
            end
        end
    end
    x_pca_aux = A_to_AA(x_pca);
    for di = 1:dims
        Y = x_pca_aux(1:di,:);
        Y_pred = y_pca(1:dims,:); %Y_pred = y_pca(1:di,:);
        r2 = corrcoef(reshape(real(L2_distance(Y,Y)),N^2,1), ...
            reshape(real(L2_distance(Y_pred,Y_pred)),N^2,1)).^2; 
        exp_p(it,di) = r2(1,2);
    end

end

end