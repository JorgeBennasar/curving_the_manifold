function PSTH = obtain_PSTH(A,targets,index_test)

PSTH = zeros(size(A,1),8,size(A,3));
for i = 1:8
    counter = 0;
    for j = 1:size(A,2)
        if targets(index_test(j)) == i
            for k = 1:size(A,1)
                PSTH(k,i,:) = PSTH(k,i,:) + A(k,j,:);
            end
            counter = counter + 1;
        end
    end
    PSTH(:,i,:) = PSTH(:,i,:)/counter;
end

% Normalization:

for i = 1:8
    PSTH(:,i,:) = PSTH(:,i,:)/max(PSTH(:,i,:),[],'all');
end

end