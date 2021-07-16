function T = transform_pls(X, pls)

W = pls.W;
P = pls.P;
T = zeros(size(X,1), size(W,2));
X = (X./ pls.xsd ) - pls.xmu;

for i = 1:size(W,2)
    w = W(:,i);
    p = P(i,:);
    T(:,i) = X * w;
    X = X - T(:,i)*p;
end

end