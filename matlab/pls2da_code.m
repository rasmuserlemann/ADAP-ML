clean;
% Test code for PLS2-DA as given by the book Chemometrics for Pattern
% Recognition. 
% PLS2 differs from PLS1 in that it uses Y matrix and can handle multiclass
% classification

datapath = 'C:\Users\csa97\Research\Projects\DuLab\ADAP-ML\adap-ml\data\';
datado = "ms";
if (datado == "ms")
X = csvread([datapath,'SCLC_study_output_filtered_2.csv'], 1, 1);
Y = [[ones(20,1);zeros(20,1)], [zeros(20,1);ones(20,1)]];
inx1 = 1:20; inx2 = 21:40;
y_true = [ones(20,1);2*ones(20,1)];
elseif (datado == "iris")
X = csvread([datapath,'fisher_iris.csv'], 1, 1);
Y = [[ones(50,1);zeros(50,1);zeros(50,1)], ...
    [zeros(50,1);ones(50,1);zeros(50,1)], ...
    [zeros(50,1);zeros(50,1);ones(50,1)]];
inx1 = 1:50; inx2 = 51:100; inx3 = 101:150;
y_true = [ones(50,1);2*ones(50,1);3*ones(50,1)];
end

[n, p] = size(X);
q = size(Y,2);
num_comp = 3;

% Get Train/Test
tr1 = randperm(20); tr2 = randperm(20)+20;
train = [tr1(1:15), tr2(1:15)]; test = [tr1(16:20), tr2(16:20)];

% Sphere Data
% {
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

totvar_X = var(reshape(X_resid, 1, numel(X_resid)));
totvar_Y = var(Y_resid);

W = zeros(p, num_comp);
T = zeros(n, num_comp);
P = zeros(num_comp, p);
Q = zeros(q, num_comp);

u = Y(:,1);
for comp = 1:num_comp
    diff = .1;
    t_old = zeros(n,1);
    iter = 0;
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
        disp(diff)
    end
    t = t_old;
    W(:,comp) = w;
    T(:,comp) = t;
    P(comp,:) = p;
    Q(:,comp) = q;
    
    X_resid = X_resid - t*p;
    Y_resid = Y_resid - t*q';
end

B = W*inv(W'*X_start'*X_start*W)*W'*X_start'*Y_start;
predict = W*inv(P*W)*Q';

resvar_X = var(reshape(X_resid, 1, numel(X_resid)));
resvar_Y = var(Y_resid);

disp("Predicted X Variance: "+100*(1-(resvar_X/totvar_X))+"%");
disp("Predicted Y Variance: "+100*(1-(resvar_Y/totvar_Y))+"%");

figure; hold on;
if(num_comp==1)
plot(inx1, T(inx1),'r')
plot(inx2, T(inx2),'b')
if(datado == "iris"); plot(inx3, T(inx3),'g'); end
hold off;
elseif(num_comp==2) 
scatter(T(inx1,1),T(inx1,2),'.','r'); 
scatter(T(inx2,1),T(inx2,2),'.','b'); 
if(datado == "iris"); scatter(T(inx3,1),T(inx3,2),'.','g'); end
hold off;
else
scatter3(T(inx1,1),T(inx1,2),T(inx1,3),'.','r'); 
scatter3(T(inx2,1),T(inx2,2),T(inx2,3),'.','b'); 
if(datado == "iris"); scatter3(T(inx3,1),T(inx3,2),T(inx3,3),'.','g'); end 
hold off;
end

% {
% Prediction:
%beta = W(:,1)*inv(P(1,:)*W(:,1))*q(1);
%beta = W*inv(P*W)*q;
y_pred = X_start*predict;
y_pred_class = zeros(n,1);
for i = 1:n
    [~, y_pred_class(i)] = max(y_pred(i,:));
end
acc = 1-sum(((y_pred_class == y_true)-1).^2)/length(y_true);
disp("Correct Classification Accuracy: "+100*acc+"%");

confusion = make_confusion_matrix(y_pred_class, y_true)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% beta is the linear transformation that predicts y %
%    To transform the data between X --> T is X*W   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;%  set('gca', 'Location', [100, 100, 800, 300]);
if(datado=="ms"); g=2; elseif(datado=="iris"); g=3; end
subplot(1,g,1); hold on;
plot(1:n, Y(:,1));
scatter(1:n, y_pred(:,1)); hold off;
legend("Actual Class Value", "Predicted Value")

subplot(1,g,2); hold on;
plot(1:n, Y(:,2));
scatter(1:n, y_pred(:,2)); hold off;
legend("Actual Class Value", "Predicted Value")

if(datado == "iris")
subplot(1,g,3); hold on;
plot(1:n, Y(:,3));
scatter(1:n, y_pred(:,3)); hold off;
legend("Actual Class Value", "Predicted Value")
end
%}