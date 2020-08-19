function d = compare_layers_procrustes(A_1,A_2,dims)

AA_1 = zeros(size(A_1,1),size(A_1,2)*size(A_1,3));
for i = 1:size(A_1,2)
    AA_1(:,(size(A_1,3)*(i-1)+1):(size(A_1,3)*i)) = squeeze(A_1(:,i,:));
end
AA_2 = zeros(size(A_2,1),size(A_2,2)*size(A_2,3));
for i = 1:size(A_2,2)
    AA_2(:,(size(A_2,3)*(i-1)+1):(size(A_2,3)*i)) = squeeze(A_2(:,i,:));
end
[A_1_pca,~,~] = get_pca(AA_1,dims);
[A_2_pca,~,~] = get_pca(AA_2,dims);
d = procrustes(transpose(A_1_pca),transpose(A_2_pca));

end
