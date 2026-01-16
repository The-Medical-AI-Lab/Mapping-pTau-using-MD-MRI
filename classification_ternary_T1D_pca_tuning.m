
clc; clear; close all;

%% 1) Load dataset and struct

load("dataset/classification_data_T1D_ternary.mat"); % expects X and yClassMean (3 unique values)
y = yClassMean; % [0.15, 0.53, 0.91]
load('dataset/DataForStat_101124.mat');
rng(0);

%% 2) Map continuous targets to class labels {0,1,2}

unique_vals = unique(y); % Extract unique values from target value
y_class = zeros(size(y));
y_class(y==unique_vals(1)) = 0;
y_class(y==unique_vals(2)) = 1;
y_class(y==unique_vals(3)) = 2;
y = y_class; % final class labels

%% 3) Compute slice boundaries for reconstruction

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

%% 4) PCA (retain 95% variance)

[coeff, score, ~, ~, explained, mu] = pca(X);
cumExpl = cumsum(explained);
numComp = find(cumExpl >= 95, 1);
fprintf('Using %d principal components (%.2f%% variance)\n', numComp, cumExpl(numComp));
X_pca = score(:, 1:numComp);
N = size(X_pca,1);

%% 5) Nested Cross-Validation: outer 5-fold, inner 5-fold for Hyperparameter Optimization

k = 5;
cv = cvpartition(y, 'KFold', k);   % stratified for multiclass

models = {'Logistic','FLD', 'SVM RBF','SVM Linear','Random Forest','MLP'};
acc   = zeros(k, numel(models));
prec  = zeros(k, numel(models));
rec   = zeros(k, numel(models));
kappa = zeros(k, numel(models));
preds = struct('logit',zeros(N,1), 'fld',zeros(N,1), 'rbf',zeros(N,1), 'lin',zeros(N,1), 'rf',zeros(N,1), 'mlp',zeros(N,1));

for fold = 1:k
    tr = training(cv, fold);
    te = test(cv, fold);
    Xtr = X_pca(tr,:);
    ytr = y(tr);
    Xte = X_pca(te,:);
    yte = y(te);

    % Inner 5-fold for hyperparameter optimization
    innerCV = cvpartition(ytr, 'KFold', 5);
    fprintf('\n==================== Fold %d/%d: Hyperparameter Tuning ====================\n', fold, k);
    
    %% (1) Logistic Regression (multinomial via ECOC)

    logit_opts = struct( ...
    'Optimizer','bayesopt', ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'CVPartition', innerCV, ...
    'MaxObjectiveEvaluations', 10, ...
    'ShowPlots', true, ...
    'Verbose', 1);

    % templateLinear with logistic learner; Lambda/Regularization 可调
    tLogit = templateLinear('Learner','logistic','Lambda',1e-4,'Regularization','ridge');
    
    logit_mdl = fitcecoc( ...
        Xtr, ytr, ...
        'Learners', tLogit, ...
        'Coding', 'onevsone', ...
        'ClassNames', [0 1 2], ...
        'OptimizeHyperparameters', {'Lambda','Regularization'}, ...
        'HyperparameterOptimizationOptions', logit_opts);
    
    fprintf('[Fold %d] Logistic best:\n', fold);
    disp(logit_mdl);
    ypred = predict(logit_mdl, Xte);
    preds.logit(te) = ypred;
    [acc(fold,5), prec(fold,5), rec(fold,5), kappa(fold,5)] = computeMetrics(yte, ypred);

    %% (2) FLD (Fisher Linear Discriminant)
    
    fld_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    fld_mdl = fitcdiscr( ...
        Xtr, ytr, ...
        'ClassNames', [0 1 2], ...
        'OptimizeHyperparameters', {'DiscrimType','Gamma','Delta'}, ...
        'HyperparameterOptimizationOptions', fld_opts);

    fprintf('[Fold %d] FLD best:\n', fold);
    disp(fld_mdl);
    ypred = predict(fld_mdl, Xte);
    preds.fld(te) = ypred;
    [acc(fold,6), prec(fold,6), rec(fold,6), kappa(fold,6)] = computeMetrics(yte, ypred);

    %% (1) SVM RBF

    svmRBF_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    tSVMrbf = templateSVM('KernelFunction','rbf','Standardize',true);
    svmRBF_mdl = fitcecoc( ...
        Xtr, ytr, ...
        'Learners', tSVMrbf, ...
        'Coding', 'onevsone', ...
        'ClassNames', [0 1 2], ...
        'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
        'HyperparameterOptimizationOptions', svmRBF_opts);

    fprintf('[Fold %d] SVM-RBF best:\n', fold);
    disp(svmRBF_mdl);
    ypred = predict(svmRBF_mdl, Xte);
    preds.rbf(te) = ypred;
    [acc(fold,1), prec(fold,1), rec(fold,1), kappa(fold,1)] = computeMetrics(yte, ypred);

    %% (2) SVM Linear

    svmLin_opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    tLinear = templateLinear('Learner','svm','Lambda',1e-4,'Regularization','ridge');
    svmLin_mdl = fitcecoc( ...
        Xtr, ytr, ...
        'Learners', tLinear, ...
        'Coding','onevsone', ...
        'ClassNames',[0 1 2], ...
        'OptimizeHyperparameters', {'Lambda','Regularization'}, ...
        'HyperparameterOptimizationOptions', svmLin_opts);

    fprintf('[Fold %d] SVM-Linear best:\n', fold);
    disp(svmLin_mdl);
    ypred = predict(svmLin_mdl, Xte);
    preds.lin(te) = ypred;
    [acc(fold,2), prec(fold,2), rec(fold,2), kappa(fold,2)] = computeMetrics(yte, ypred);

    %% (3) Random Forest

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
        'ClassNames', [0 1 2], ...
        'OptimizeHyperparameters', {'NumLearningCycles','MinLeafSize','MaxNumSplits'}, ...
        'HyperparameterOptimizationOptions', rf_opts);

    fprintf('[Fold %d] RandomForest best:\n', fold);
    disp(rf_mdl);
    ypred = predict(rf_mdl, Xte);
    preds.rf(te) = ypred;
    [acc(fold,3), prec(fold,3), rec(fold,3), kappa(fold,3)] = computeMetrics(yte, ypred);

    %% (4) MLP

    mlp_opts = struct( ...
        'Optimizer', 'bayesopt', ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 1);

    mlp_mdl = fitcnet( ...
        Xtr, ytr, ...
        'ClassNames', [0 1 2], ...
        'Standardize', true, ...
        'IterationLimit', 500, ...
        'OptimizeHyperparameters', {'NumLayers','Activations','Layer_1_Size','Layer_2_Size','Layer_3_Size'}, ...
        'HyperparameterOptimizationOptions', mlp_opts);

    fprintf('[Fold %d] MLP best:\n', fold);
    disp(mlp_mdl);
    ypred = predict(mlp_mdl, Xte);
    preds.mlp(te) = ypred;
    [acc(fold,4), prec(fold,4), rec(fold,4), kappa(fold,4)] = computeMetrics(yte, ypred);
end

%% 6) Display CV results

fprintf('\n===== 5-Fold CV Performance (mean ± std) =====\n');
fprintf('Model         Acc(μ±σ)     Prec(μ±σ)    Rec(μ±σ)     Kappa(μ±σ)\n');
for m = 1:numel(models)
    fprintf('%-12s  %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', ...
        models{m}, mean(acc(:,m)), std(acc(:,m)), mean(prec(:,m)), std(prec(:,m)), mean(rec(:,m)), std(rec(:,m)), mean(kappa(:,m)), std(kappa(:,m)));
end

%% 7) Plot metrics

acc_mean = mean(acc);
acc_std = std(acc);
prec_mean = mean(prec);
prec_std = std(prec);
rec_mean = mean(rec);
rec_std = std(rec);
kappa_mean = mean(kappa);
kappa_std = std(kappa);

fig = figure; set(fig, 'Position', [100, 100, 1300, 800]);

subplot(2,2,1);
bar(acc_mean);
hold on;
errorbar(1:numel(models), acc_mean, acc_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', models);
ylabel('Accuracy');
ylim([0 1]);
title('Accuracy');
grid on;

subplot(2,2,2);
bar(prec_mean);
hold on;
errorbar(1:numel(models), prec_mean, prec_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', models);
ylabel('Precision');
ylim([0 1]);
title('Precision');
grid on;

subplot(2,2,3);
bar(rec_mean);
hold on;
errorbar(1:numel(models), rec_mean, rec_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', models);
ylabel('Recall');
ylim([0 1]);
title('Recall');
grid on;

subplot(2,2,4);
bar(kappa_mean);
hold on;
errorbar(1:numel(models), kappa_mean, kappa_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', models);
ylabel('Cohen''s Kappa');
ylim([0 1]);
title('Cohen''s Kappa');
grid on;

sgtitle('5-Fold CV Performance Metrics for T1D (Ternary) under PCA');
saveas(fig, 'image/classification ternary/performance_crossvalidation_T1D_ternary_pca_tuning.png');

%% 8) Reconstruct classification maps (map back 0/1/2 -> original unique values)

label_map = unique_vals;
fields = {'logit','fld','rbf','lin','rf','mlp'};

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
            img(rows(j),cols(j)) = label_map(lbls(j)+1);
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
    fname = sprintf('image/classification ternary/reconstruction_%s_T1D_ternary_pca_tuning.png', lower(models{m}));
    saveas(gcf, fname);
end

%% 10) Explainability: PCA back-projection Grad-CAM-like importance maps

outDir = 'image/explainability';
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% (A) Logistic: PC-importance

fprintf('\n[Explainability] Computing PC-importance for Logistic Regression\n');
w_pc_logit = getPCImportanceFromECOCLinear(logit_mdl, numComp); % [numComp x 1]
w_orig_logit = coeff(:,1:numComp) * w_pc_logit; % [d_orig x 1], [2500, 1]

% Normalization
w_orig_logit_n = w_orig_logit - min(w_orig_logit);
if max(w_orig_logit_n) > 0
    w_orig_logit_n = w_orig_logit_n / max(w_orig_logit_n);
end

% Save vector importance
csvwrite(fullfile(outDir, 'importance_logistic_origspace_T1D.csv'), w_orig_logit);

% Visualization
figL = figure('Units','inches','Position',[1 1 6 5]);
imagesc(reshape(w_orig_logit_n, [side, side]));
axis image off;
colormap('jet');
colorbar;
title(sprintf('T1D Grad-CAM-like Importance (Logistic Regression)', numComp, side, side));
saveas(figL, fullfile(outDir, sprintf('heatmap_logistic_backproject_T1D.png', side, side)));

%% (B) FLD: PC-importance

fprintf('[Explainability] Computing PC-importance for FLD (pairwise Coeffs)...\n');
classNames = fld_mdl.ClassNames;
K = numel(classNames);

w_pc_fld = zeros(numComp,1);
cnt = 0;

for iIdx = 1:K
    for jIdx = iIdx+1:K
        s = fld_mdl.Coeffs(iIdx, jIdx);
        v = [];
        
        % Linear discriminant
        if isstruct(s) && isfield(s, 'Linear') && ~isempty(s.Linear)
            v = abs(s.Linear(:));
        end

        % Secondary discriminant
        if isstruct(s) && isfield(s, 'Quadratic') && ~isempty(s.Quadratic)
            q = s.Quadratic;
            vq = sum(abs(q), 2); % [numComp x 1]
            if isempty(v)
                v = vq;
            else
                v = v + vq;
            end
        end

        if ~isempty(v)
            k = min(numComp, numel(v));
            w_pc_fld(1:k) = w_pc_fld(1:k) + v(1:k);
            cnt = cnt + 1;
        end
    end
end

if cnt > 0
    w_pc_fld = w_pc_fld / cnt;
else
    warning('FLD: No usable pairwise coefficients found (check DiscrimType and Coeffs).');
end

w_orig_fld = coeff(:,1:numComp) * w_pc_fld; % [d_orig x 1]
w_orig_fld_n = w_orig_fld - min(w_orig_fld);
if max(w_orig_fld_n) > 0, w_orig_fld_n = w_orig_fld_n / max(w_orig_fld_n); end

csvwrite(fullfile(outDir, 'importance_fld_origspace_T1D.csv'), w_orig_fld);

figF = figure('Units','inches','Position',[1 1 6 5]);
imagesc(reshape(w_orig_fld_n, [side, side]));
axis image off;
colormap('jet');
colorbar;
title(sprintf('T1D Grad-CAM-like Importance (FLD)', numComp, side, side));
saveas(figF, fullfile(outDir, sprintf('heatmap_fld_backproject_T1D.png', side, side)));

%% SVM with linear and rbf kernels



%% Other Models

d_orig = size(X, 2); % 2500
side = round(sqrt(d_orig));
is_square = (side * side == d_orig);

fprintf('\n[Explainability] Generating Grad-CAM-like maps for T1D (Ternary)\n');

modelsExplain = { ...
    struct('mdl',svmRBF_mdl,'name','SVM_RBF'), ...
    struct('mdl',svmLin_mdl,'name','SVM_Linear'), ...
    struct('mdl',rf_mdl,'name','RandomForest'), ...
    struct('mdl',mlp_mdl,'name','MLP')};

for mi = 1:numel(modelsExplain)
    mdl = modelsExplain{mi}.mdl;
    name = modelsExplain{mi}.name;
    fprintf('  > %s ...\n', name);
    
    % Compute permutation importance on PCA features
    imp = oobPermutedPredictorImportanceSafe(mdl, X_pca, y, numComp);
    if isempty(imp)
        fprintf('[Warn] Model type unsupported for permutation importance.\n');
        continue;
    end
    
    % Back-project
    w_orig = coeff(:,1:numComp) * imp(:);
    w_orig_n = normalizeTo01(w_orig);
    writematrix(w_orig, fullfile(outDir, sprintf('importance_%s_origspace_T1D.csv', lower(name))));
    
    if is_square
        fig = figure('Units','inches','Position',[1 1 6 5]);
        imagesc(reshape(w_orig_n,[side,side]));
        axis image off; colormap('jet'); colorbar;
        title(sprintf('Grad-CAM-like Importance (%s)\n%d PCs → %dx%d', name, numComp, side, side));
        saveas(fig, fullfile(outDir, sprintf('heatmap_%s_backproject_T1D.png', lower(name))));
    end
end

fprintf('[Explainability] Done. Saved all maps under %s\n', outDir);

%% 9) Helper: compute metrics

function [ac, pr, re, ka] = computeMetrics(yT, yP)
    C = confusionmat(yT, yP);
    ac = sum(diag(C)) / max(sum(C(:)),1);

    % Macro precision/recall with zero-division guards
    prec_k = diag(C) ./ max(sum(C,1)',1);   % per-class precision
    reca_k = diag(C) ./ max(sum(C,2),1);    % per-class recall
    pr = mean(prec_k);
    re = mean(reca_k);

    % Cohen's Kappa (multiclass)
    N = sum(C(:));
    Po = ac;
    Pe = sum(sum(C,1).*sum(C,2)) / max(N^2,1);
    ka = (Po - Pe) / max(1 - Pe, eps);
end
