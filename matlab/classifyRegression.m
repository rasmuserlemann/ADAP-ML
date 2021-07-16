function y_pred = classifyRegression(y)

[n, ~] = size(y);

y_pred = zeros(n,1);
for i = 1:n
    y_pred(i) = find(y(i,:) == max(y(i,:)));
end

end