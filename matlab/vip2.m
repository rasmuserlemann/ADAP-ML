function vip = vip2(pls_model)

W = pls_model.W;
T = pls_model.T;
Q = pls_model.Q;

[~, h] = size(T);
p = size(W,1);

VIP = zeros(h, p);

s=diag(T'*T*Q'*Q);

for i = 1:p
    weight = [];
    for j=1:h
        weight(j,1)= (W(i,j)/norm(W(:,j)))^2;
    end
    q=s.*weight;  % explained variance by variable i
    
    VIP(:,i) = sqrt(p*q/sum(s));
end

vip = VIP;

end