function [Y_pred, A] = LSTM_predict(X, param, r_or_c, amp_noise, noise_samples)

% X: input data, shape: (n_input, m_test, t_time)
% param: struct containing the following:
    % 1) W_f: weight matrix of forget gate, shape: (n_hidden, n_hidden + n_input)
    % 2) b_f: bias of the forget gate, shape: (n_hidden, 1)
    % 3) W_i: weight matrix of the update gate, shape: (n_hidden, n_hidden + n_input)
    % 4) b_i: bias of the update gate, shape: (n_hidden, 1)
    % 5) W_c: weigth matrix of the first "tanh", shape: (n_hidden, n_hidden + n_input)
    % 6) b_c: bias of the first "tanh", shape: (n_hidden, 1)
    % 7) W_o: weight matrix of the output gate, shape: (n_hidden, n_hidden + n_input)
    % 8) b_o: bias of the output gate, shape: (n_hidden, 1)
    % 9) W_y: weigth matrix relating hidden state to output, shape: (n_output, n_hidden)
    % 10) b_y: bias relating hidden state to output, shape: (n_output, 1)
    % 11) G: weighted connections matrix
% r_or_c: 'regression' or 'classification'
% amp_noise = training noise amplitude
% noise_samples = training noise samples

[~, m_test, t_time] = size(X);
n_hidden = size(param.W_f,1);

%{

noise = zeros(n_hidden,t_time);
for i_n = 1:n_hidden
    j_n = round(rand*(length(param.coloured_noise)-1)+1);
    noise(i_n,:) = param.coloured_noise(j_n,:);
end

%}

coloured_noise = amp_noise*noise_samples;

noise = zeros(n_hidden,t_time);
for i_n = 1:n_hidden
    j_n = round(rand*(size(noise_samples,1)-1)+1);
    noise(i_n,:) = coloured_noise(j_n,:);
end
        
[A, Y_aux, ~] = LSTM_forward_prop(X, param, r_or_c, noise);

if strcmp(r_or_c,'regression')
    Y_pred = Y_aux;
elseif strcmp(r_or_c,'classification')
    Y_pred = zeros(size(Y_aux));
    for i = 1:m_test
        idx = find(Y_aux(:,i)==max(Y_aux(:,i)));
        Y_pred(idx,i) = 1;
    end
end

end