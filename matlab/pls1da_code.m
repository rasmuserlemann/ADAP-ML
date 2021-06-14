clean;
% Test code for PLS1-DA. The algorithm here
% follows that outlined in:
% Loong Chuen Lee, Choong-Yeun Liong, Abdul Aziz Jemain. 
% Partial least squares-discriminant analysis (PLS-DA) for classification of 
% high-dimensional (HD) data: a review of contemporary practice strategies 
% and knowledge gaps.
% Analyst, 2018,143, 3526-3539

datapath = 'C:\Users\csa97\Research\Projects\DuLab\ADAP-ML\adap-ml\data\';
X = csvread([datapath,'SCLC_study_output_filtered_2.csv'], 1, 1);
Y = csvread([datapath,'SCLC_study_responses_2.csv'], 1, 1);
Yd = [[ones(20,1);zeros(20,1)], [zeros(20,1);ones(20,1)]];

inx1 = 1:20; inx2 = 21:40;

[n, p] = size(X);
q = size(Y,2);
num_comp = 3;

% Get Train/Test
tr1 = randperm(20); tr2 = randperm(20)+20;
train = [tr1(1:15), tr2(1:15)]; test = [tr1(16:20), tr2(16:20)];

% Sphere Data
% {
sd_x = std(X, [], 1); 
sd_y = std(Y); 

X_std = zeros(size(X));
for i = 1:p; X_std(:,i) = X(:,i)/sd_x(i); end
Y_std = Y/sd_y;
%}

% Mean Center
mu_x = mean(X_std,1);
mu_y = mean(Y_std,1);

X_start = X_std - mu_x;
Y_start = Y_std - mu_y;

X_resid = X_start;
Y_resid = Y_start;

totvar_X = var(reshape(X_resid, 1, numel(X_resid)));
totvar_Y = var(Y_resid);

W = zeros(p, num_comp);
T = zeros(n, num_comp);
P = zeros(num_comp, p);
Q = zeros(1, num_comp);
resid_var = zeros(2,num_comp);
for comp = 1:num_comp
    w = X_resid'*Y_resid; % (p x n) * (n x 1) = (p x 1)
    w = w/norm(w);
    t = X_resid*w; % (n x p) * (p x 1) = (n x 1)
    tmp2 = norm(t)^2;
    p = t'*X_resid / tmp2; % (1 x n) * (n x p) = (1 x p)
    q = Y_resid'*t / tmp2; % (1 x n) * (n x 1) = (1 x 1)
    
    % Save Component Info
    W(:,comp) = w;
    T(:,comp) = t;
    P(comp,:) = p;
    Q(comp) = q;
    
    % Calculate Residuals
    X_resid = X_resid - t*p;
    Y_resid = Y_resid - t*q;
    
    resid_var(1, comp) = var(reshape(X_resid, 1, numel(X_resid)))/totvar_X;
    resid_var(2, comp) = var(Y_resid)/totvar_Y;
end

if(num_comp>1)
figure; hold on; 
scatter(T(inx1,1),T(inx1,2),'.','r'); 
scatter(T(inx2,1),T(inx2,2),'.','b'); hold off;
end

% Prediction:
%beta = W(:,1)*inv(P(1,:)*W(:,1))*q(1);
beta = W*inv(P*W)*Q';
y_pred = X_start*beta;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% beta is the linear transformation that predicts y %
%    To transform the data between X --> T is X*W   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure; hold on;
plot(1:40, Y(:,1));
scatter(1:40, y_pred(:,1)); hold off;
legend("Actual Class Value", "Predicted Value")
