function model = pls2da(X, Y, num_comp)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to compute Partial Least Squares Discriminant Analysis         %
% This code follows the PLS2 algorithm which treats classes as a dummy    %
% index                                                                   %
% The NIPALS (Non-Linear Iterative Partial Least Squares) algorithm is    %
% followed in order to compute the PLS model.                             %
%                                                                         %
% INPUTS:                                                                 %
%         X = Matrix of data observations (n observations x p variables)  %
%         Y = Dummy matrix of class labels (n observations x c classes)   %
%         num_comp = Number of components to calculate                    %
% OUTPUTS:                                                                %
%         model = MATLAB struct containing the informational matricies    %
%         model.xmu = original feature means                              %
%         model.ymu = original class means                                %
%         model.xsd = original feature standard deviations                %
%         model.ysd = original class standard deviations                  %
%         model.W = PLS Weights that transform a sample to latent space   %
%         model.T = X Scores for training data in X                       %
%         model.P = X Loadings to transfrom T back to X                   %
%         model.Q = Y Loadings to predict Y from T                        %
%         model.beta = matrix for predicting new class label (beta = WQ') %
%         model.stats = [% X Variance Predicted, % Y Variance Predicted]  %
%                                                                         %
% PLS MODEL RELATIONSHIPS:                                                %
%         X = T*P  + E                                                    %
%         Y = T*Q' + F                                                    %
%         T = X_k*W_k (X_k is the kth residual data matrix, W_k is the    %
%                                                    kth weight vector    %
%         Y_Predicted = T*Q'                                              %
%                                                                         %
% From SKLEARN CODE:                                                      %
%         T = X * (W*inv(P*W)); % Non-Iterative transfromation to latent  %
%                                                                  space  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n, p] = size(X);
q = size(Y,2);

% Scale Data
sd_x = std(X, [], 1); 
sd_y = std(Y, [], 1); 

X_std = zeros(size(X));
Y_std = zeros(size(Y));
for i = 1:p; X_std(:,i) = X(:,i)/sd_x(i); end
for i = 1:q; Y_std(:,i) = Y(:,i)/sd_y(i); end
%}

% Mean Center
mu_x = mean(X_std,1);
mu_y = mean(Y_std,1);

X_start = X_std - mu_x;
Y_start = Y_std - mu_y;

X_resid = X_start;
Y_resid = Y_start;

% Compute total variance for statistical anlaysis
totvar_X = var(reshape(X_resid, 1, numel(X_resid)));
totvar_Y = var(Y_resid);

% Begin NIPALS
W = zeros(p, num_comp); % Weights
T = zeros(n, num_comp); % Scores
P = zeros(num_comp, p); % X Loadings
Q = zeros(q, num_comp); % Y Loadings

u = Y(:,1); % This can be arbitrarily selected
for comp = 1:num_comp
    diff = .1;
    t_old = zeros(n,1);
    iter = 1;
    while(diff > .01)
        w = X_resid'*u;
        w = w/norm(w);
        t_new = X_resid*w;
        p = t_new'*X_resid / norm(t_new)^2;
        q = Y_resid'*t_new / norm(t_new)^2;
        if(iter == 1)
            u = Y*q/norm(q);
            t_old = t_new;
        else
            err = t_old - t_new;
            diff = sum(sum(err.^2));
            t_old = t_new;
        end
        iter = iter + 1;
        %disp(diff)
    end
    t = t_old;
    W(:,comp) = w;
    T(:,comp) = t;
    P(comp,:) = p;
    Q(:,comp) = q;
    
    X_resid = X_resid - t*p;
    Y_resid = Y_resid - t*q';
end
% End NIPALS

resvar_X = var(reshape(X_resid, 1, numel(X_resid)));
resvar_Y = var(Y_resid);
stats = [1-(resvar_X/totvar_X), 1-(resvar_Y/totvar_Y)];

%disp("Predicted X Variance: "+100*(1-(resvar_X/totvar_X))+"%");
%disp("Predicted Y Variance: "+100*(1-(resvar_Y/totvar_Y))+"%");

% Put out put model together!
model.xmu = mu_x;
model.xsd = sd_x;
model.ymu = mu_y;
model.ysd = sd_y;
model.W = W;
model.T = T;
model.P = P;
model.Q = Q;
model.trans = W*inv(P*W);
model.beta = model.trans*Q';
model.stats = stats;

transform = @(X) ((X./sd_x)-mu_x) * model.trans;
predict = @(X) ((X./sd_x)-mu_x) * model.beta; 

model.transform = transform;
model.predict = predict;

end