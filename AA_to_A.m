function A = AA_to_A(AA,time)

A = zeros(size(AA,1),size(AA,2)/time,time);
for i = 1:size(AA,2)/time
    for j = 1:time
        A(:,i,j) = AA(:,time*(i-1)+j);
    end
end

end