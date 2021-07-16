function Y = predict_pls(X, pls)

Y = transform_pls(X,pls)*pls.Q';

end