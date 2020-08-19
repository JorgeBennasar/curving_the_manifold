function AA = A_to_AA(A)

AA = zeros(size(A,1),size(A,2)*size(A,3));
for i = 1:size(A,2)
    AA(:,(size(A,3)*(i-1)+1):(size(A,3)*i)) = squeeze(A(:,i,:));
end

end