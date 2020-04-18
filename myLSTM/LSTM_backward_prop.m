function grad_hidden = LSTM_backward_prop(X, dA, cache, param, r_or_c)

% X: input data, shape: (n_input, m_trials, t_time)
% dA: gradients of hidden states, shape: (n_hidden, m_trials, t_time)
% cache: struct of t_time dimensions containing the following:
    % 1) A_next
    % 2) A_prev
    % 3) C_next
    % 4) C_prev
    % 5) Ft
    % 6) It
    % 7) Cct
    % 8) Ot
% param: struct containing the following:
    % 1) W_f: weight matrix of forget gate, shape: (n_hidden, n_hidden + n_input)
    % 2) b_f: bias of the forget gate, shape: (n_hidden, 1)
    % 3) W_i: weight matrix of the update gate, shape: (n_hidden, n_hidden + n_input)
    % 4) b_i: bias of the update gate, shape: (n_hidden, 1)
    % 5) W_c: weigth matrix of the first "tanh", shape: (n_hidden, n_hidden + n_input)
    % 6) b_c: bias of the first "tanh", shape: (n_hidden, 1)
    % 7) W_o: weight matrix of the output gate, shape: (n_hidden, n_hidden + n_input)
    % 8) b_o: bias of the output gate, shape: (n_hidden, 1)
    % 5) W_y: weigth matrix relating hidden state to output, shape: (n_output, n_hidden)
    % 5) b_y: bias relating hidden state to output, shape: (n_output, 1)
% r_or_c: 'regression' or 'classification'

[n_hidden, m_trials, ~] = size(dA);
[~, ~, t_time] = size(X);

dW_f = zeros(size(param.W_f));
dW_i = zeros(size(param.W_i));
dW_c = zeros(size(param.W_c));
dW_o = zeros(size(param.W_o));
db_f = zeros(size(param.b_f));
db_i = zeros(size(param.b_i));
db_c = zeros(size(param.b_c));
db_o = zeros(size(param.b_o));

if strcmp(r_or_c,'regression')
    dA_prev = zeros(n_hidden,m_trials);
    dC_prev = zeros(n_hidden,m_trials);
elseif strcmp(r_or_c,'classification')
    dA_prev = dA;
    dC_prev = zeros(size(dA));
end
    
for t = t_time:-1:1
    if strcmp(r_or_c,'regression')
        dA_next = dA_prev + dA(:,:,t);
    elseif strcmp(r_or_c,'classification')
        dA_next = dA_prev;
    end
    dC_next = dC_prev;

    A_prev = cache(t).A_prev;
    C_next = cache(t).C_next;
    C_prev = cache(t).C_prev;
    Ft = cache(t).Ft;
    It = cache(t).It;
    Cct = cache(t).Cct;
    Ot = cache(t).Ot;

    dot = dA_next.*tanh(C_next);
    dcct = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*It;
    dit = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*Cct;
    dft = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*C_prev;

    dit = dit.*It.*(1-It);
    dft = dft.*Ft.*(1-Ft);
    dot = dot.*Ot.*(1-Ot);
    dcct = dcct.*(1-Cct.^2);

    X_t = X(:,:,t); 
    concat = [A_prev; X_t];

    dW_f = dW_f + dft*transpose(concat);
    dW_i = dW_i + dit*transpose(concat);
    dW_c = dW_c + dcct*transpose(concat);
    dW_o = dW_o + dot*transpose(concat);
    db_f = db_f + sum(dft,2);
    db_i = db_i + sum(dit,2);
    db_c = db_c + sum(dcct,2);
    db_o = db_o + sum(dot,2);

    d_concat = transpose(param.W_f)*dft + transpose(param.W_o)*dot + ...
        transpose(param.W_i)*dit + transpose(param.W_c)*dcct;
    dA_prev = d_concat(1:n_hidden,:);
    dC_prev = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*Ft;
end

grad_hidden.dW_f = dW_f;
grad_hidden.dW_i = dW_i;
grad_hidden.dW_c = dW_c;
grad_hidden.dW_o = dW_o;
grad_hidden.db_f = db_f;
grad_hidden.db_i = db_i;
grad_hidden.db_c = db_c;
grad_hidden.db_o = db_o;
  
end