function vip = vip3(pls_model, X, Yd)

W = pls_model.W;
Q = pls_model.Q;
T = pls_model.transform(X);
mu_y = pls_model.ymu;
sd_y = pls_model.ysd;
Yd = (Yd./sd_y) - mu_y;

%X = (X./sd_x)-mu_x;
sst = sum(sum(Yd.^2));

[~, cp] = size(W);
[~, df] = size(X);

vip_t = zeros(cp, df);
for i = 1:cp
    w = W(:,i);
    q = Q(i);
    t = T(:,1);
    y_pred = t*q;
    ssr = sum(sum(y_pred.^2));
    vip_t(i,:) = w'.^2*ssr/sst;
end

vip_t = sqrt(df*vip_t/cp);
vip = sqrt(df*sum(vip_t,1)/cp);

end