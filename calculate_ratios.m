function [ratios,num] = calculate_ratios(N,iter,trials,sp)

if size(N,1) > sp
    d = fix(size(N,1)/sp);
    ratios = zeros(1,d);
    num = zeros(1,d);
    for i = 1:d
        disp([num2str(i) ' / ' num2str(d)]);
        num(i) = i*sp;
        [m_i,~,m_p,~] = compare_performance(N,iter,i*sp,trials);
        ratios(i) = calculate_PR(m_i)/calculate_PR(m_p);
    end
else
    ratios = 0;
    num = 0;
    disp('Not sufficient number of units');
end

end
        
        

