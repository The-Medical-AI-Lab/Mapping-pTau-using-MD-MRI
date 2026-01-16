
clc; clear; close all;

%% 1) Load dataset and struct

load("dataset/classification_data_T1D_binary.mat"); % X and yBinary
y = yBinary;
load('dataset/DataForStat_101124.mat'); % DataForStat for reconstruction
rng(0);

%% 2) Compute slice boundaries for reconstruction

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

%% 3) PCA (retain 95% variance)

[coeff, score, ~, ~, explained, mu] = pca(X);
cumExpl = cumsum(explained);
numComp = find(cumExpl >= 95, 1);
fprintf('Using %d principal components (%.2f%% variance)\n', numComp, cumExpl(numComp));
X_pca = score(:, 1:numComp);
N = size(X_pca,1);

%% 4) 5-Fold Cross-Validation (outer): train, tune (inner), predict, metrics

% Define 5-fold cross-validation partition.
% cvpartition for classification keeps class balance per fold.
k = 5;
cv = cvpartition(y, 'KFold', k);

% Preallocate metrics and full-sample predictions
acc   = struct('log',zeros(k,1),'fld',zeros(k,1),'rbf',zeros(k,1),'lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));
prec  = struct('log',zeros(k,1),'fld',zeros(k,1),'rbf',zeros(k,1),'lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));
rec   = struct('log',zeros(k,1),'fld',zeros(k,1),'rbf',zeros(k,1),'lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));
kappa = struct('log',zeros(k,1),'fld',zeros(k,1),'rbf',zeros(k,1),'lin',zeros(k,1),'rf',zeros(k,1),'mlp',zeros(k,1));

% Full-length prediction holders (one label per voxel for each model)
preds = struct('log',zeros(N,1),'fld',zeros(N,1),'rbf',zeros(N,1),'lin',zeros(N,1),'rf',zeros(N,1),'mlp',zeros(N,1));

% For printing best hyperparameters per fold
models = {'Logistic','FLD','SVM_RBF','SVM_Linear','RandomForest','MLP'};

for fold = 1:k

     tr = training(cv, fold);
    te = test(cv, fold);
    Xtr = X_pca(tr,:);
    ytr = y(tr);
    Xte = X_pca(te,:);
    yte = y(te);

    % Inner 5-fold CV for hyperparameter optimization (nested CV)
    innerCV = cvpartition(ytr, 'KFold', 5);

    fprintf('\n==================== Fold %d/%d: Hyperparameter Tuning ====================\n', fold, k);

    %% (1) Logistic Regression

    % Tuned hyperparameters:
    % Lambda: L2 regularization strength (larger => more regularization)
    % Regularization: 'ridge' (L2), 'lasso' (L1), or 'elasticnet' (combination)

    log_opts = struct( ... % Hyperparameter Optimization Options
        'Optimizer','bayesopt', ... % Bayesian Optimization
        'AcquisitionFunctionName','expected-improvement-plus', ... % Acquisition Function
        'CVPartition', innerCV, ... % Inner Cross-validation
        'Repartition', false, ... % Fix the same innerCV
        'MaxObjectiveEvaluations', 10, ... % The maximum number of different sets of hyperparameters attempted
        'ShowPlots', true, ... % Real-time covergence plot plotted during Bayesian Optimization
        'Verbose', 1); % Setting the level of output detail

    log_mdl = fitclinear( ... % Applying fitclinear to train logistic regression
        Xtr, ytr, ... % Feature Matrix and Target Value
        'Learner','logistic', ... % Set the learner for logistic regression
        'ClassNames',[0 1], ... % Binary classification and the target is 0 and 1
        'OptimizeHyperparameters', {'Lambda','Regularization'}, ... % Hyperparameters: Lambda, Regularization
        'HyperparameterOptimizationOptions', log_opts); % Hyperparameter Optimization Options

    % Print best hyperparameters
    fprintf('[Fold %d] Logistic best: ', fold);
    disp(log_mdl);

    % Predict
    ypred = predict(log_mdl, Xte);
    preds.log(te) = ypred;
    [acc.log(fold), prec.log(fold), rec.log(fold), kappa.log(fold)] = computeMetrics(yte, ypred);
    disp([acc.log(fold), prec.log(fold), rec.log(fold), kappa.log(fold)])
    fprintf('\n')

    %% (2) Fisher Linear Discriminant

    % Tuned hyperparameters:
    % Gamma: Tikhonov (ridge) regularization added to covariance (stabilizes inverse)
    % Delta: Shrinkage of linear coefficients toward zero (additional regularization)

    fld_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    fld_mdl = fitcdiscr( ...
        Xtr, ytr, ...
        'DiscrimType','linear', ...
        'ClassNames',[0 1], ...
        'OptimizeHyperparameters', {'Gamma','Delta'}, ... % Hyperparameters: Gamma, Delta
        'HyperparameterOptimizationOptions', fld_opts);

    fprintf('[Fold %d] FLD best: ', fold);
    disp(fld_mdl);

    ypred = predict(fld_mdl, Xte);
    preds.fld(te) = ypred;
    [acc.fld(fold), prec.fld(fold), rec.fld(fold), kappa.fld(fold)] = computeMetrics(yte, ypred);
    disp([acc.fld(fold), prec.fld(fold), rec.fld(fold), kappa.fld(fold)])
    fprintf('\n')

    %% (3) SVM RBF

    % Tuned hyperparameters:
    % BoxConstraint: Soft-margin C (larger => lower bias, higher variance)
    % KernelScale: RBF width parameter (gamma = 1/(2*KernelScale^2))

    svmRBF_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    svmRBF_mdl = fitcsvm( ...
        Xtr, ytr, ...
        'KernelFunction','rbf', ...
        'Standardize', true, ...
        'ClassNames',[0 1], ...
        'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ... % Hyperparameters: BoxConstraint, KernelScale
        'HyperparameterOptimizationOptions', svmRBF_opts);

    fprintf('[Fold %d] SVM-RBF best: ', fold);
    disp(svmRBF_mdl);

    ypred = predict(svmRBF_mdl, Xte);
    preds.rbf(te) = ypred;
    [acc.rbf(fold), prec.rbf(fold), rec.rbf(fold), kappa.rbf(fold)] = computeMetrics(yte, ypred);
    disp([acc.rbf(fold), prec.rbf(fold), rec.rbf(fold), kappa.rbf(fold)])
    fprintf('\n')

    %% (4) SVM Linear

    % Tuned hyperparameters:
    % BoxConstraint: Soft-margin C

    svmLin_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    svmLin_mdl = fitclinear( ...
        Xtr, ytr, ...
        'Learner','svm', ...
        'ClassNames',[0 1], ...
        'OptimizeHyperparameters', {'Lambda','Regularization'}, ... % Hyperparameters: Lambda, Regularization
        'HyperparameterOptimizationOptions', svmLin_opts);

    fprintf('[Fold %d] SVM-Linear best: ', fold);
    disp(svmLin_mdl);

    ypred = predict(svmLin_mdl, Xte);
    preds.lin(te) = ypred;
    [acc.lin(fold), prec.lin(fold), rec.lin(fold), kappa.lin(fold)] = computeMetrics(yte, ypred);
    disp([acc.lin(fold), prec.lin(fold), rec.lin(fold), kappa.lin(fold)])
    fprintf('\n')

    %% (5) Random Forest

    % Tuned hyperparameters:
    % NumLearningCycles: Number of trees
    % MinLeafSize: Minimum leaf size (controls tree depth / variance)
    % MaxNumSplits: Max number of decision splits per tree (another depth control)

    rf_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    rf_mdl = fitcensemble( ...
        Xtr, ytr, ...
        'Method','Bag', ...
        'Learners','Tree', ...
        'ClassNames',[0 1], ...
        'OptimizeHyperparameters', {'NumLearningCycles','MinLeafSize','MaxNumSplits'}, ... % Hyperparameters: NumLearningCycles, MinLeafSize, MaxNumSplits
        'HyperparameterOptimizationOptions', rf_opts);

    fprintf('[Fold %d] RandomForest best: ', fold);
    disp(rf_mdl);

    ypred = predict(rf_mdl, Xte);
    preds.rf(te) = ypred;
    [acc.rf(fold), prec.rf(fold), rec.rf(fold), kappa.rf(fold)] = computeMetrics(yte, ypred);
    disp([acc.rf(fold), prec.rf(fold), rec.rf(fold), kappa.rf(fold)])
    fprintf('\n')

    %% (6) MLP

    % Tuned hyperparameters:
    % LayerSizes: Hidden layer width (vector allowed, here 1 hidden layer candidates)
    % Activations: Nonlinearity (e.g., 'relu','tanh','sigmoid'); affects capacity & trainability
    % IterationLimit: Max training iterations (early stopping may also apply internally)
    % Standardize: Whether to z-score features (recommended for MLP)

    mlp_opts = struct( ...
        'Optimizer', 'bayesopt', ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    mlp_mdl = fitcnet(Xtr, ytr, ...
        'ClassNames',[0 1], ...
        'Standardize', true, ...
        'IterationLimit', 1000, ...
        'OptimizeHyperparameters', {'NumLayers','Activations','Layer_1_Size','Layer_2_Size','Layer_3_Size'}, ...
        'HyperparameterOptimizationOptions', mlp_opts);

    fprintf('[Fold %d] MLP best: ', fold);
    disp(mlp_mdl);

    ypred = predict(mlp_mdl, Xte);
    preds.mlp(te) = ypred;
    [acc.mlp(fold), prec.mlp(fold), rec.mlp(fold), kappa.mlp(fold)] = computeMetrics(yte, ypred);
    disp([acc.mlp(fold), prec.mlp(fold), rec.mlp(fold), kappa.mlp(fold)])
    fprintf('\n')

end

%% 5) Display CV results

metrics     = [mean(acc.log), std(acc.log); mean(prec.log), std(prec.log); mean(rec.log), std(rec.log); mean(kappa.log), std(kappa.log)];
metricsFld  = [mean(acc.fld), std(acc.fld); mean(prec.fld), std(prec.fld); mean(rec.fld), std(rec.fld); mean(kappa.fld), std(kappa.fld)];
metricsRbf  = [mean(acc.rbf), std(acc.rbf); mean(prec.rbf), std(prec.rbf); mean(rec.rbf), std(rec.rbf); mean(kappa.rbf), std(kappa.rbf)];
metricsLin  = [mean(acc.lin), std(acc.lin); mean(prec.lin), std(prec.lin); mean(rec.lin), std(rec.lin); mean(kappa.lin), std(kappa.lin)];
metricsRf   = [mean(acc.rf),  std(acc.rf);  mean(prec.rf),  std(prec.rf);  mean(rec.rf),  std(rec.rf);  mean(kappa.rf),  std(kappa.rf)];
metricsMlp  = [mean(acc.mlp), std(acc.mlp); mean(prec.mlp), std(prec.mlp); mean(rec.mlp), std(rec.mlp); mean(kappa.mlp), std(kappa.mlp)];

fprintf('\n===== 5-Fold CV Performance (mean ± std) =====\n');
fprintf('Model          Acc(μ±σ)     Prec(μ±σ)    Rec(μ±σ)     Kappa(μ±σ)\n');
fprintf('Logistic     : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', metrics');
fprintf('FLD          : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', metricsFld');
fprintf('SVM (RBF)    : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', metricsRbf');
fprintf('SVM (Linear) : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', metricsLin');
fprintf('Random Forest: %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', metricsRf');
fprintf('MLP          : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', metricsMlp');

%% 6) Plot all metrics

acc_mean   = [mean(acc.log), mean(acc.fld), mean(acc.rbf), mean(acc.lin), mean(acc.rf), mean(acc.mlp)];
prec_mean  = [mean(prec.log), mean(prec.fld), mean(prec.rbf), mean(prec.lin), mean(prec.rf), mean(prec.mlp)];
rec_mean   = [mean(rec.log),  mean(rec.fld),  mean(rec.rbf),  mean(rec.lin),  mean(rec.rf),  mean(rec.mlp)];
kappa_mean = [mean(kappa.log), mean(kappa.fld), mean(kappa.rbf), mean(kappa.lin), mean(kappa.rf), mean(kappa.mlp)];

acc_std   = [std(acc.log), std(acc.fld), std(acc.rbf), std(acc.lin), std(acc.rf), std(acc.mlp)];
prec_std  = [std(prec.log), std(prec.fld), std(prec.rbf), std(prec.lin), std(prec.rf), std(prec.mlp)];
rec_std   = [std(rec.log),  std(rec.fld),  std(rec.rbf),  std(rec.lin),  std(rec.rf),  std(prec.mlp)]; % note: last one uses prec.mlp? fix to kappa earlier
rec_std(end) = std(rec.mlp); % fix typo
kappa_std = [std(kappa.log), std(kappa.fld), std(kappa.rbf), std(kappa.lin), std(kappa.rf), std(kappa.mlp)];

model_labels = {'Logistic', 'FLD', 'SVM RBF', 'SVM Linear', 'RF', 'MLP'};

fig = figure;
set(fig, 'Position', [100, 100, 1300, 800]);

subplot(2,2,1);
bar(acc_mean);
hold on;
errorbar(1:numel(acc_mean), acc_mean, acc_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Accuracy'); ylim([0 1]); title('Accuracy'); grid on;

subplot(2,2,2);
bar(prec_mean);
hold on;
errorbar(1:numel(prec_mean), prec_mean, prec_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Precision'); ylim([0 1]); title('Precision'); grid on;

subplot(2,2,3);
bar(rec_mean);
hold on;
errorbar(1:numel(rec_mean), rec_mean, rec_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Recall'); ylim([0 1]); title('Recall'); grid on;

subplot(2,2,4);
bar(kappa_mean);
hold on;
errorbar(1:numel(kappa_mean), kappa_mean, kappa_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Cohen''s Kappa'); ylim([0 1]); title('Cohen''s Kappa'); grid on;

sgtitle('5-Fold CV Performance Metrics for T1D under PCA');
saveas(fig, 'image/classification binary/performance_crossvalidation_T1D_pca_tuning.png');

%% 7) Reconstruct & plot classification maps

fields = {'log','fld','rbf','lin','rf','mlp'};
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
        title(DataForStat.cases{i}, 'FontName', 'Arial', 'FontSize', 16);
    end
    % sgtitle(sprintf('Reconstructed - %s', strrep(models{m},'_',' ')),'FontName','Arial');
    fname = sprintf('image/classification binary/reconstruction_%s_T1D_pca_tuning.png', lower(models{m}));
    saveas(gcf, fname);
end

%% Helper: compute metrics

function [ac, pr, re, ka] = computeMetrics(yT, yP)
    C = confusionmat(yT, yP);
    % guard against degenerate cases
    if numel(C)==1
        if yT(1)==0
            C = [C(1),0;0,0];
        else
            C = [0,0;0,C(1)];
        end
    end
    TN = C(1,1); FP = C(1,2); FN = C(2,1); TP = C(2,2);
    ac = (TP+TN)/sum(C(:));
    pr = TP / max(TP+FP, eps);
    re = TP / max(TP+FN, eps);
    Po = (TP+TN)/sum(C(:));
    Pe = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(sum(C(:))^2);
    ka = (Po-Pe) / max(1-Pe, eps);
end
