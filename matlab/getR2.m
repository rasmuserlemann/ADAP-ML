function r2 = getR2(y_pred, y_true)

mu = mean(y_true);

press = sum(sum((y_true-y_pred).^2));
tss = sum(sum((y_true-mu).^2));

r2 = 1-(press/tss);

end