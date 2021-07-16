function vip = getVIP(pls_model, Y)

W = pls_model.W;
T = pls_model.T;

[p, n_comp] = size(W);
n_samp = size(T,1);

r = corr2(T,Y);

vip = zeros(p, n_comp);
vip(:,1) = W(:,1).^2;

if(n_comp > 1)
    for i = 2:n_comp
        R = sum(r(1:i,:),2);
        vip(:,i) = (R' * (W(:,1:i).^2)') / sum(R);
    end
end

vip = sqrt(p*vip);

end