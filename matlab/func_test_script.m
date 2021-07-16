clean;
%Script for Testing PLS Functions
do = "MS_data";

%% Load Data
datapath = 'C:\Users\csa97\Research\Projects\DuLab\ADAP-ML\adap-ml\data\';
if (do == "MS_data")
    X = csvread([datapath,'SCLC_study_output_filtered_2.csv'], 1, 1);
    Y = csvread([datapath,'SCLC_study_responses_2.csv'], 1, 1)+1;
    Yd = [[ones(20,1);zeros(20,1)], [zeros(20,1);ones(20,1)]];
    inx = {1:20, 21:40};
    var=[295,464,961,1000,1076,1078,1150,1153,1162,1256,1259,...
        1262,1276,1283,1365,1381,1387,1414,1553];
elseif (do == "IRIS_data")
    X = csvread([datapath,'fisher_iris.csv'], 1, 1);
    Y = [ones(50,1);2*ones(50,1);3*ones(50,1)];
    Yd = [[ones(50,1);zeros(50,1);zeros(50,1)], ...
        [zeros(50,1);ones(50,1);zeros(50,1)], ...
        [zeros(50,1);zeros(50,1);ones(50,1)]];
    inx = {1:50, 51:100, 101:150};
    var=["Sepal Length","Sepal Width","Petal Length","Petal Width"];
end

%% Analysis
% Build Model
pls_model = pls2da(X, Yd, 3);
y_pred = classifyRegression(pls_model.T*pls_model.Q');

% Get R2
R2 = getR2(y_pred, Y);

% Get VIP
vip = getVIP(pls_model,Y);
%vip = vip2(pls_model)';
%vip = vip3(pls_model, X, Y)';


%% Plotting
% Plot
figure; hold on;
for i = 1:length(inx)
scatter(pls_model.T(inx{i},1), pls_model.T(inx{i},2), 'filled')
end; hold off;
xlabel("PLS Mode 1"); ylabel("PLS Mode 2"); title("PLS Projections");

if (do == "MS_data")
    figure;
    r_vip = load("mixOmics_vip.txt");
    sim_vip = load("simca_vip.txt");
    adap_vip = load("adap_vip.txt");
    rmse = sqrt(mean((vip-r_vip).^2, 1));
    N = 1;
    for i = 1:N%size(vip,2)
        subplot(N,1,i);%size(vip,2),1, i); 
        hold on; plot(vip(:,i)); % MATLAB
        plot(r_vip(:,i), '--'); % R
        plot(sim_vip(:,i), ':'); % SIMCA
        plot(adap_vip(:,i), '.-'); % ADAPML
        xticks(1:19); xticklabels(var); xtickangle(90);
        title("VIP Mode "+i+": rmse="+rmse(i)); 
    end
    legend("MATLAB", "R","SIMCA","ADAP-ML");
end