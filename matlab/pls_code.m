clean;
% Test code for PLS Regression. The algorithm here follows that outlined
% in:
% A MULTIVARIATE CALIBRATION PROBLEM IN ANALYTICAL CHEMISTRY SOLVED BY
% PARTIAL LEAST-SQUARES MODELS IN LATENT VARIABLES (1983)
% MICHAEL SJGSTRGM and SVANTE WOLD, WALTER LINDBERG and JAN-kE PERSSON, 
% HARALD MARTENS

datapath = 'C:\Users\csa97\Research\Projects\DuLab\ADAP-ML\adap-ml\data\';
X = csvread([datapath,'SCLC_study_output_filtered_2.csv'], 1, 1);
Y = csvread([datapath,'SCLC_study_responses_2.csv'], 1, 1);

[n, p] = size(X);
q = size(Y,2);

% The authors recommend variance scaling as a first step if nothing is
% known about the variates prior to the analysis
sd_x = std(X, [], 1); 
sd_y = std(Y); 

for i = 1:p; X(:,i) = X(:,i)/sd_x(i); end
Y = Y/sd_y;

% The next step is mean centering both X and Y
mu_x = mean(X,1);
mu_y = mean(Y);

% The authors call these the residual matrix
e = X - mu_x;
f = Y - mu_y;

% A PLS model attempts to create 2 models, one for the X data and one for
% the Y data via matrix decomposition:
% X = mu_x + B*T + E
% Y = my_y + B*U + F
% In the above, B is the PLS loadings or components (consisting of the
% column space), T and U are the latent space varables for each sample in X
% and Y respectively, and E and F are the error in the representation for X
% and Y respectively

% Lets find 2 pls components by setting
A = 1;
U = zeros(n, A);
T = zeros(n, A);
Bx = zeros(A, p);
By = zeros(A, q);
W = zeros(A, p);

% Start
for i = 1:q; U(:, 1) = f(:,1)/norm(f(:,1)); end

% For the X block
W = U'*e;
T = e*W'; 
for i = 1:A; T(:,i) = T(:,i)/norm(T(:,i)); end

% For the Y block
By = T'*f;
U = f*By';

% Outter Relation:
Bx = T'*e;
% Inner Relation:
C = U'*T;

% Calculate Residuals
X_resid = e - T*Bx;
Y_resid = f - C*T*By; 
