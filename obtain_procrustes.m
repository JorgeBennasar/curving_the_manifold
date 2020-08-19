function d = obtain_procrustes(A,dims)

AA = zeros(size(A,1),size(A,2)*size(A,3));
for i = 1:size(A,2)
    AA(:,(size(A,3)*(i-1)+1):(size(A,3)*i)) = squeeze(A(:,i,:));
end
[AA_pca,~,~] = get_pca(AA,dims);
[AA_isomap,~] = get_isomap(AA,dims);
if strcmp(AA_isomap,'Not computed')
    disp('Error: isomap cannot compute matrix');
    d = 'Not computed';
else
    d = procrustes(transpose(AA_pca),transpose(AA_isomap));
end

end