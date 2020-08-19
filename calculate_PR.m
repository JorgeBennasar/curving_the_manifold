function PR = calculate_PR(f)
 
eig = zeros(size(f));
for i = 1:length(f)
    if i == 1
        eig(i) = f(i);
    else
        eig(i) = f(i) - f(i-1);
    end
end
PR = sum(eig)^2/sum(eig.^2);

end