
clc; clear; close all;

%% 1. Load dataset and struct

load("dataset/regression_data_T2D.mat"); % X, y
load('dataset/DataForStat_101124.mat'); % For reconstruction
rng(0);

%% 2. Compute slice boundaries for reconstruction

nSlices = numel(DataForStat.bwMaskLowFinal);
sliceStarts = zeros(nSlices,1);
sliceEnds   = zeros(nSlices,1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

%% 3. PCA (retain 95% variance)

[coeff, score, ~, ~, explained, mu] = pca(X);
cumExpl = cumsum(explained);
numComp = find(cumExpl >= 95, 1);
fprintf('Using %d principal components (%.2f%% variance)\n', numComp, cumExpl(numComp));
X_pca = score(:, 1:numComp);
N = size(X_pca,1);

%% 4. 5-Fold CV: outer loop

k = 5;
cv = cvpartition(N, 'KFold', k);

mse = struct('lin',zeros(k,1),'poly',zeros(k,1),'svm_rbf',zeros(k,1),'svm_lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));
rsq = struct('lin',zeros(k,1),'poly',zeros(k,1),'svm_rbf',zeros(k,1),'svm_lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));
preds = struct('lin',zeros(k,1),'poly',zeros(k,1),'svm_rbf',zeros(k,1),'svm_lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));

models = {'Linear','Polynomial','SVM RBF','SVM Linear','Random Forest','MLP'};
fields = {'lin','poly','svm_rbf','svm_lin','rf','mlp'};

% Nested Cross-validation
for fold = 1:k
    tr = training(cv, fold);
    te = test(cv, fold);
    Xtr = X_pca(tr,:);
    ytr = y(tr);
    Xte = X_pca(te,:);
    yte = y(te);

    innerCV = cvpartition(ytr, 'KFold', 5); % inner tuning

    fprintf('\n================ Fold %d/%d: Hyperparameter Tuning ================\n', fold, k);

    %% (1) Linear Regression

    lin_opts = struct('Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition',innerCV, ...
        'MaxObjectiveEvaluations',10, ...
        'ShowPlots',true, ...
        'Verbose',1);

    mdl_lin = fitrlinear( ...
        Xtr,ytr, ...
        'Learner','leastsquares', ...
        'OptimizeHyperparameters',{'Lambda','Regularization'}, ...
        'HyperparameterOptimizationOptions',lin_opts);

    fprintf('[Fold %d] Linear best:\n', fold);
    disp(mdl_lin.ModelParameters);

    ypred = predict(mdl_lin,Xte);
    preds.lin(te) = ypred;
    [mse.lin(fold), rsq.lin(fold)] = computeRegressionMetrics(yte, ypred);
    disp([mse.lin(fold), rsq.lin(fold)])
    fprintf('\n')

    %% (2) Polynomial Regression (degree fixed=2, tune regularization)

    Xtr_poly = [Xtr, Xtr.^2];
    Xte_poly = [Xte, Xte.^2];
    poly_opts = struct('Optimizer','bayesopt', ...
        'CVPartition',innerCV,...
        'MaxObjectiveEvaluations',10, ...
        'ShowPlots',true, ...
        'Verbose',1);

    mdl_poly = fitrlinear(Xtr_poly,ytr, ...
        'Learner','leastsquares', ...
        'OptimizeHyperparameters',{'Lambda','Regularization'}, ...
        'HyperparameterOptimizationOptions',poly_opts);

    fprintf('[Fold %d] Polynomial best:\n', fold);
    disp(mdl_poly.ModelParameters);

    ypred = predict(mdl_poly,Xte_poly);
    preds.poly(te) = ypred;
    [mse.poly(fold), rsq.poly(fold)] = computeRegressionMetrics(yte, ypred);
    disp([mse.poly(fold), rsq.poly(fold)])
    fprintf('\n')

    %% (3) SVM Regression (RBF)

    svmRBF_opts = struct('Optimizer','bayesopt', ...
        'CVPartition',innerCV, ...
        'MaxObjectiveEvaluations',10, ...
        'ShowPlots',true, ...
        'Verbose',1);

    mdl_svm_rbf = fitrsvm(Xtr,ytr, ...
        'KernelFunction','rbf', ...
        'Standardize',true, ...
        'OptimizeHyperparameters',{'BoxConstraint','KernelScale','Epsilon'}, ...
        'HyperparameterOptimizationOptions',svmRBF_opts);

    fprintf('[Fold %d] SVM-RBF best:\n', fold);
    disp(mdl_svm_rbf.ModelParameters);

    ypred = predict(mdl_svm_rbf,Xte);
    preds.svm_rbf(te) = ypred;
    [mse.svm_rbf(fold), rsq.svm_rbf(fold)] = computeRegressionMetrics(yte, ypred);
    disp([mse.svm_rbf(fold), rsq.svm_rbf(fold)])
    fprintf('\n')

    %% (4) SVM Regression (Linear)

    svmLin_opts = struct('Optimizer','bayesopt', ...
        'CVPartition',innerCV, ...
        'MaxObjectiveEvaluations',10, ...
        'ShowPlots',true, ...
        'Verbose',1);

    mdl_svm_lin = fitrsvm(Xtr,ytr, ...
        'KernelFunction','linear', ...
        'Standardize',true, ...
        'OptimizeHyperparameters',{'BoxConstraint','Epsilon'}, ...
        'HyperparameterOptimizationOptions',svmLin_opts);

    fprintf('[Fold %d] SVM-Linear best:\n', fold);
    disp(mdl_svm_lin.ModelParameters);

    ypred = predict(mdl_svm_lin,Xte);
    preds.svm_lin(te) = ypred;
    [mse.svm_lin(fold), rsq.svm_lin(fold)] = computeRegressionMetrics(yte, ypred);
    disp([mse.svm_lin(fold), rsq.svm_lin(fold)])
    fprintf('\n')

    %% (5) Random Forest Regression

    rf_opts = struct('Optimizer','bayesopt', ...
        'CVPartition',innerCV, ...
        'MaxObjectiveEvaluations',10, ...
        'ShowPlots',true, ...
        'Verbose',1);

    mdl_rf = fitrensemble(Xtr,ytr, ...
        'Method','Bag', ...
        'OptimizeHyperparameters',{'NumLearningCycles','MinLeafSize','MaxNumSplits'},...
        'HyperparameterOptimizationOptions',rf_opts);

    fprintf('[Fold %d] RF best:\n', fold);
    disp(mdl_rf.ModelParameters);

    ypred = predict(mdl_rf,Xte);
    preds.rf(te) = ypred;
    [mse.rf(fold), rsq.rf(fold)] = computeRegressionMetrics(yte, ypred);
    disp([mse.rf(fold), rsq.rf(fold)])
    fprintf('\n')

    %% (6) MLP Regression

    mlp_opts = struct('Optimizer','bayesopt', ...
        'CVPartition',innerCV, ...
        'MaxObjectiveEvaluations',10, ...
        'ShowPlots',true, ...
        'Verbose',1);

    mdl_mlp = fitrnet(Xtr,ytr, ...
        'Standardize',true, ...
        'IterationLimit',1000,...
        'OptimizeHyperparameters',{'LayerSizes','Activations'},...
        'HyperparameterOptimizationOptions',mlp_opts);

    fprintf('[Fold %d] MLP best:\n', fold);
    disp(mdl_mlp.ModelParameters);

    ypred = predict(mdl_mlp,Xte);
    preds.mlp(te) = ypred;
    [mse.mlp(fold), rsq.mlp(fold)] = computeRegressionMetrics(yte, ypred);
    disp([mse.mlp(fold), rsq.mlp(fold)])
    fprintf('\n')

end

%% 5. Display CV results

for i = 1:numel(models)
    m = fields{i};
    fprintf('%s Regression: MSE=%.3f±%.3f, R²=%.3f±%.3f\n', ...
        models{i}, mean(mse.(m)), std(mse.(m)), mean(rsq.(m)), std(rsq.(m)));
end

%% 6. Plot metrics

mse_mean = cellfun(@(f) mean(mse.(f)), fields);
rsq_mean = cellfun(@(f) mean(rsq.(f)), fields);
mse_std  = cellfun(@(f) std(mse.(f)), fields);
rsq_std  = cellfun(@(f) std(rsq.(f)), fields);

fig = figure;
set(fig, 'Position', [100, 100, 1000, 600]);

subplot(1,2,1);
bar(mse_mean);
hold on;
errorbar(1:6,mse_mean,mse_std,'.k','LineWidth',1.2);
hold off;
set(gca,'XTickLabel',models);
ylabel('MSE');
title('Mean Squared Error');
grid on;

subplot(1,2,2);
bar(rsq_mean);
hold on;
errorbar(1:6,rsq_mean,rsq_std,'.k','LineWidth',1.2);
hold off;
set(gca,'XTickLabel',models);
ylabel('R²');
title('R-Squared');
grid on;

sgtitle('5-Fold CV Regression Performance (with tuning) - T2D');
saveas(fig,'image/regression/performance_crossvalidation_T2D_pca_tuning.png');

%% 7. Reconstruct maps

for m = 1:numel(fields)
    field = fields{m};
    result = cell(1,nSlices);
    for i = 1:nSlices
        mask = DataForStat.bwMaskLowFinal{i};
        img  = zeros(size(mask));
        lbls = preds.(field)(sliceStarts(i):sliceEnds(i));
        rows = DataForStat.row_pixCurFinal{i};
        cols = DataForStat.col_pixCurFinal{i};
        for j = 1:numel(lbls)
            img(rows(j),cols(j)) = lbls(j);
        end
        result{i} = img;
    end
    figure('Units','inches','Position',[1 1 24 5]);
    for i = 1:nSlices
        subplot(1,nSlices,i);
        imagesc(result{i});
        colormap('jet');
        colorbar;
        clim([0 1]);
        axis off;
        title(DataForStat.cases{i},'FontName','Arial');
    end
    sgtitle(sprintf('Reconstructed - %s Regression - T2D', models{m}),'FontName','Arial');
    fname = sprintf('image/regression/reconstruction_%s_T2D_pca_tuning.png',lower(models{m}));
    saveas(gcf,fname);
end

%% Helper

function [mse_val, rsq_val] = computeRegressionMetrics(yT,yP)
    mse_val = mean((yT - yP).^2);
    ss_res = sum((yT - yP).^2);
    ss_tot = sum((yT - mean(yT)).^2);
    rsq_val = 1 - ss_res/ss_tot;
end
