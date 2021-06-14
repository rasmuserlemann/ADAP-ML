function r = corr2(A, B)

[m1, n] = size(A);
[m2, c] = size(B);

if(m1 ~= m2)
    error("Inconsistent number of samples in matrices");
end

a_mu = mean(A,1); a_sd = std(A, [], 1);
b_mu = mean(B,1); b_sd = std(B, [], 1);

r = zeros(n,c);

for i = 1:n
    a = (A(:,i) - a_mu(i))/a_sd(i);
    for j = 1:c
        b = (B(:,j) - b_mu(j))/b_sd(j);
        r(i,j) = a'*b/(m1-1);
    end
end

end