
clear; clc; close all;

%% Paths

trainFile = "dataset/regression_data_T1D.mat";
testFile = "external dataset/regression_data_T1D_external.mat";
metaFile = "external dataset/DataForStat_120825.mat";
outDir = "image/external regression";
modelOutDir = "model/model_external";
resultMat = fullfile(modelOutDir, "external_results_all_models_T1D.mat");

%% Load train and test

S_tr = load(trainFile);
S_te = load(testFile);

Xtr = S_tr.X;
Xte = S_te.X;

ytr = pickContinuousTarget(S_tr,'training');
yte = pickContinuousTarget(S_te,'external');

ytr = ytr(:);
yte = yte(:);

fprintf('Loaded training: X=%s, y=%s\n', mat2str(size(Xtr)), mat2str(size(ytr)));
fprintf('Loaded external: X=%s, y=%s\n', mat2str(size(Xte)), mat2str(size(yte)));

%% PCA (>=95%%)

[coeff, ~, ~, ~, explained, mu] = pca(Xtr);
kPCA = find(cumsum(explained) >= 95, 1, 'first'); % k principal components
fprintf('PCA: using %d PCs (%.2f%% variance)\n', kPCA, sum(explained(1:kPCA)));

Xtr_pca = (Xtr - mu) * coeff(:, 1:kPCA);
Xte_pca = (Xte - mu) * coeff(:, 1:kPCA);

% Print sizes before / after PCA
fprintf('Xtr original size: %d × %d\n', size(Xtr,1), size(Xtr,2));
fprintf('Xtr PCA size: %d × %d\n', size(Xtr_pca,1), size(Xtr_pca,2));
fprintf('Xte original size: %d × %d\n', size(Xte,1), size(Xte,2));
fprintf('Xte PCA size: %d × %d\n', size(Xte_pca,1), size(Xte_pca,2));

%% Model list (Regression)

models = { ...
    struct('name','Linear','field','lin'), ...
    struct('name','Polynomial2','field','poly2'), ...
    struct('name','SVM RBF','field','svm_rbf'), ...
    struct('name','SVM Linear','field','svm_lin'), ...
    struct('name','Random Forest','field','rf'), ...
    struct('name','MLP','field','mlp') ...
};

% Store results
R = struct();
R.trainFile = trainFile;
R.testFile = testFile;
R.metaFile = metaFile;
R.kPCA = kPCA;
R.explained = explained;
R.mu = mu;
R.coeff = coeff;

%% Hyperparameter tuning on TRAIN only (5-fold CV)

% we perform Bayesian optimization with 5-fold cross-validation within the training set to find the optimal hyperparameters
% we retrain the final model on the full training set using those optimal hyperparameters, and finally evaluate it on the external test set.

innerCV = cvpartition(numel(ytr), 'KFold', 5);

for mi = 1:numel(models)
    name = models{mi}.name;
    field = models{mi}.field;

    fprintf('\n==================== [%s] Tuning on TRAIN ====================\n', name);

    switch field

        %% Linear regression
        case 'lin'
            
            lin_opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_lin = fitrlinear( ...
                Xtr_pca,ytr, ...
                'Learner','leastsquares', ...
                'OptimizeHyperparameters',{'Lambda','Regularization'}, ...
                'HyperparameterOptimizationOptions',lin_opts);

            % retrain final on full train with best params
            best = mdl_lin.ModelParameters;
            % disp(best)
            mdl_final = fitrlinear( ...
                Xtr_pca, ytr, ...
                'Learner','leastsquares', ...
                'Lambda', best.Lambda, ...
                'Regularization', best.Regularization);

            yhat_tr = predict(mdl_final, Xtr_pca);
            [rmse_tr, mae_tr, r2_tr, r_tr] = computeRegMetrics(ytr, yhat_tr);
            fprintf('[%s] Training (resubstitution): MSE=%.4f, R^2=%.4f\n', name, rmse_tr^2, r2_tr);
            
            R.(field).rmse_tr = rmse_tr;
            R.(field).r2_tr = r2_tr;

        %% Polynomial regression degree=2
        case 'poly2'
            
            Xtr_poly = [Xtr_pca, Xtr_pca.^2];
            Xte_poly = [Xte_pca, Xte_pca.^2];

            poly_opts = struct('Optimizer','bayesopt', ...
                'CVPartition',innerCV,...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);
        
            mdl_poly = fitrlinear(Xtr_poly,ytr, ...
                'Learner','leastsquares', ...
                'OptimizeHyperparameters',{'Lambda','Regularization'}, ...
                'HyperparameterOptimizationOptions',poly_opts);

            % retrain final on full train with best params
            best = mdl_poly.ModelParameters;
            % disp(best)
            mdl_final = fitrlinear( ...
                Xtr_poly, ytr, ...
                'Learner','leastsquares', ...
                'Lambda', best.Lambda, ...
                'Regularization', best.Regularization);
            yhat_tr = predict(mdl_final, Xtr_poly);
            [rmse_tr, mae_tr, r2_tr, r_tr] = computeRegMetrics(ytr, yhat_tr);
            fprintf('[%s] Training (resubstitution): MSE=%.4f, R^2=%.4f\n', name, rmse_tr^2, r2_tr);
                        
            R.(field).rmse_tr = rmse_tr;
            R.(field).r2_tr = r2_tr;

        %% SVM Linear
        case 'svm_lin'

            svmLin_opts = struct('Optimizer','bayesopt', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);
        
            mdl_svm_lin = fitrsvm(Xtr_pca,ytr, ...
                'KernelFunction','linear', ...
                'Standardize',true, ...
                'OptimizeHyperparameters',{'BoxConstraint','Epsilon'}, ...
                'HyperparameterOptimizationOptions',svmLin_opts);

            best = mdl_svm_lin.ModelParameters;
            % disp(best)
            mdl_final = fitrsvm( ...
                Xtr_pca, ytr, ...
                'KernelFunction','linear', ...
                'Standardize', true, ...
                'BoxConstraint', best.BoxConstraint, ...
                'Epsilon', best.Epsilon);

            yhat_tr = predict(mdl_final, Xtr_pca);
            [rmse_tr, mae_tr, r2_tr, r_tr] = computeRegMetrics(ytr, yhat_tr);
            fprintf('[%s] Training (resubstitution): MSE=%.4f, R^2=%.4f\n', name, rmse_tr^2, r2_tr);     

            R.(field).rmse_tr = rmse_tr;
            R.(field).r2_tr = r2_tr;

        %% SVM RBF
        case 'svm_rbf'

            svmRBF_opts = struct('Optimizer','bayesopt', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);
        
            mdl_svm_rbf = fitrsvm(Xtr_pca,ytr, ...
                'KernelFunction','rbf', ...
                'Standardize',true, ...
                'OptimizeHyperparameters',{'BoxConstraint','KernelScale','Epsilon'}, ...
                'HyperparameterOptimizationOptions',svmRBF_opts);

            % final uses best params
            best = mdl_svm_rbf.ModelParameters;
            % disp(best)
            mdl_final = fitrsvm( ...
                Xtr_pca, ytr, ...
                'KernelFunction','rbf', ...
                'Standardize', true, ...
                'BoxConstraint', best.BoxConstraint, ...
                'KernelScale', best.KernelScale, ...
                'Epsilon', best.Epsilon);

            yhat_tr = predict(mdl_final, Xtr_pca);
            [rmse_tr, mae_tr, r2_tr, r_tr] = computeRegMetrics(ytr, yhat_tr);
            fprintf('[%s] Training (resubstitution): MSE=%.4f, R^2=%.4f\n', name, rmse_tr^2, r2_tr);
                        
            R.(field).rmse_tr = rmse_tr;
            R.(field).r2_tr = r2_tr;

        %% Random forest regression
        case 'rf'

            rf_opts = struct('Optimizer','bayesopt', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);
        
            mdl_rf = fitrensemble(Xtr_pca,ytr, ...
                'Method','Bag', ...
                'OptimizeHyperparameters',{'NumLearningCycles','MinLeafSize','MaxNumSplits'},...
                'HyperparameterOptimizationOptions',rf_opts);

            best = mdl_rf.HyperparameterOptimizationResults.XAtMinObjective;
            % disp(best)
            tTree = templateTree( ...
                'MaxNumSplits', best.MaxNumSplits, ...
                'MinLeafSize', best.MinLeafSize);
            mdl_final = fitrensemble( ...
                Xtr_pca, ytr, ...
                'Method','Bag', ...
                'Learners', tTree, ...
                'NumLearningCycles', best.NumLearningCycles);

            yhat_tr = predict(mdl_final, Xtr_pca);
            [rmse_tr, mae_tr, r2_tr, r_tr] = computeRegMetrics(ytr, yhat_tr);
            fprintf('[%s] Training (resubstitution): MSE=%.4f, R^2=%.4f\n', name, rmse_tr^2, r2_tr);
                        
            R.(field).rmse_tr = rmse_tr;
            R.(field).r2_tr = r2_tr;

        %% MLP Regression    
        case 'mlp'

            mlp_opts = struct('Optimizer','bayesopt', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);
        
            mdl_mlp = fitrnet(Xtr_pca,ytr, ...
                'Standardize',true, ...
                'IterationLimit',1000,...
                'OptimizeHyperparameters',{'LayerSizes','Activations'},...
                'HyperparameterOptimizationOptions',mlp_opts);

            best = mdl_mlp.ModelParameters;
            % disp(best)
            mdl_final = fitrnet( ...
                Xtr_pca, ytr, ...
                'Standardize', true, ...
                'IterationLimit', 1000, ...
                'LayerSizes',  best.LayerSizes, ...
                'Activations', best.Activations);

            yhat_tr = predict(mdl_final, Xtr_pca);
            [rmse_tr, mae_tr, r2_tr, r_tr] = computeRegMetrics(ytr, yhat_tr);
            fprintf('[%s] Training (resubstitution): MSE=%.4f, R^2=%.4f\n', name, rmse_tr^2, r2_tr);
                        
            R.(field).rmse_tr = rmse_tr;
            R.(field).r2_tr = r2_tr;

        otherwise
            error('Unknown model field: %s', field);
    end

    % External prediction
    fprintf('[%s] Predicting on EXTERNAL\n', name);

    if strcmp(field,'poly2')
        yhat = predict(mdl_final, Xte_poly);
    else
        yhat = predict(mdl_final, Xte_pca);
    end

    % External metrics
    [rmse, mae, r2, r] = computeRegMetrics(yte, yhat);
    fprintf('[%s] External: MSE=%.6f, R^2=%.6f\n', name, rmse^2, r2);

    % save per-model result
    R.(field).name = name;
    R.(field).rmse = rmse;
    % R.(field).mae = mae;
    R.(field).r2 = r2;
    % R.(field).r = r;
    R.(field).yhat = yhat;

    % save model file
    % modelFile = fullfile(modelOutDir, sprintf('final_%s_T1D_external.mat', field));
    % save(modelFile, 'mdl_final');
    % fprintf('Saved model: %s\n', modelFile);

    % scatter plot
    figS = figure('Units','inches','Position',[1 1 6 6]);
    plot(yte, yhat, '.', 'MarkerSize', 8);
    hold on;
    minv = min([yte(:); yhat(:)]);
    maxv = max([yte(:); yhat(:)]);
    plot([minv maxv],[minv maxv],'k-','LineWidth',1);
    xlabel('Ground Truth');
    ylabel('Prediction');
    title(sprintf('%s (External) (R^2=%.3f, r=%.3f)', name, r2, r));
    axis square;
    grid on;
    % exportgraphics(figS, fullfile(outDir, sprintf('scatter_%s_external.png', field)), 'Resolution', 300);
    % close(figS);

    set(figS, 'PaperPositionMode', 'auto');
    print(figS, fullfile(outDir, sprintf('scatter_%s_external.png', field)), '-dpng', '-r600');
    close(figS);

end

%% Training Summary
fprintf('\n================== TRAINING Summary ==================\n');

fields = cellfun(@(s) s.field, models, 'UniformOutput', false);
names = cellfun(@(s) s.name,  models, 'UniformOutput', false);

for i = 1:numel(fields)
    f = fields{i};
    fprintf('%-14s | MSE = %.6f | R^2 = %.4f\n', names{i}, R.(f).rmse_tr.^2, R.(f).r2_tr);
end

mse_tr_all = zeros(numel(fields),1);
r2_tr_all  = zeros(numel(fields),1);

for i = 1:numel(fields)
    f = fields{i};
    mse_tr_all(i) = R.(f).rmse_tr.^2;
    r2_tr_all(i) = R.(f).r2_tr;
end

figTr = figure('Units','inches','Position',[1 1 12 5]);

% MSE
subplot(1,2,1);
bar(mse_tr_all);
set(gca,'XTickLabel',names,'FontSize',11);
xtickangle(30);
ylabel('Mean Squared Error');
title('Training Set MSE');
grid on;

% R^2
subplot(1,2,2);
bar(r2_tr_all);
set(gca,'XTickLabel',names,'FontSize',11);
xtickangle(30);
ylabel('R^2');
ylim([0 1]);
title('Training Set R^2');
grid on;

sgtitle('Training Performance Summary (T1D Regression)');

% Save
set(figTr,'PaperPositionMode','auto');
print(figTr, fullfile(outDir,'training_metrics_all_models.png'), '-dpng','-r600');
close(figTr);

%% Summary bar plot (external metrics)

fields = cellfun(@(s) s.field, models, 'UniformOutput', false);
names = cellfun(@(s) s.name,  models, 'UniformOutput', false);

mse_all = zeros(numel(fields),1);
% rmse_all = zeros(numel(fields),1);
% mae_all = zeros(numel(fields),1);
r2_all = zeros(numel(fields),1);
% r_all = zeros(numel(fields),1);

for i = 1:numel(fields)
    f = fields{i};
    mse_all(i) = R.(f).rmse.^2;
    % rmse_all(i) = R.(f).rmse;
    % mae_all(i) = R.(f).mae;
    r2_all(i) = R.(f).r2;
    % r_all(i) = R.(f).r;
end

figM = figure('Units','inches','Position',[1 1 12 6]);

% subplot(2,2,1); bar(rmse_all); set(gca,'XTickLabel',names); title('RMSE'); grid on; xtickangle(30);
% subplot(2,2,2); bar(mae_all); set(gca,'XTickLabel',names); title('MAE'); grid on; xtickangle(30);
% subplot(2,2,3); bar(r2_all); set(gca,'XTickLabel',names); title('R^2'); grid on; xtickangle(30);
% subplot(2,2,4); bar(r_all); set(gca,'XTickLabel',names); title('r'); grid on; xtickangle(30);

subplot(1,2,1);
bar(mse_all);
set(gca,'XTickLabel',names);
title('MSE');
ylabel('Mean Squared Error');
grid on;
xtickangle(30);

subplot(1,2,2);
bar(r2_all);
set(gca,'XTickLabel',names);
title('R^2');
ylabel('Coefficient of Determination');
grid on;
xtickangle(30);

sgtitle('External Test Metrics (T1D Regression)');

set(figM, 'PaperPositionMode', 'auto');
print(figM, fullfile(outDir,'external_metrics_all_models.png'), '-dpng', '-r600');
close(figM);

%% Load external DataForStat for reconstruction

M = load(metaFile);
if isfield(M,'DataForStat2')
    DataForStat = M.DataForStat2;
else
    error('Expected DataForStat2 in %s', metaFile);
end

% make sure indices matches how you created the external dataset
indices = 1:6;
[sliceStarts, sliceEnds] = computeSlicePartitions(DataForStat, indices);

assert(sliceEnds(end) == numel(yte), 'Partition mismatch: sum slice voxels (%d) != length(yte) (%d). Check indices/order.', sliceEnds(end), numel(yte));

% choose consistent clim: use [0,1] (your pTau range), or use data-driven:
useFixed01 = true;

%% Reconstruct maps: Predicted vs GroundTruth for each model

for i = 1:numel(fields)
    f = fields{i};
    yhat = R.(f).yhat(:);

    % pred/true package
    preds = struct();
    preds.pred = yhat;
    preds.true = yte;

    % color limits
    if useFixed01
        cl = [0 1];
    else
        allVals = [preds.pred(:); preds.true(:)];
        cl = [min(allVals) max(allVals)];
        if ~isfinite(cl(1)) || ~isfinite(cl(2)) || cl(1)==cl(2)
            cl = [min(preds.pred) max(preds.pred)];
        end
    end

    makeReconPanel(DataForStat, indices, sliceStarts, sliceEnds, preds.pred, cl, ...
        fullfile(outDir, sprintf('recon_%s_pred_external.png', f)), ...
        sprintf('%s - Predicted (external)', R.(f).name));

    makeReconPanel(DataForStat, indices, sliceStarts, sliceEnds, preds.true, cl, ...
        fullfile(outDir, sprintf('recon_%s_true_external.png', f)), ...
        sprintf('%s - GroundTruth (external)', R.(f).name));
end

%% Helper functions

function y = pickContinuousTarget(S, whichSet)
    cand = {'yReg','y','yContinuous','yTarget'};
    y = [];
    for k = 1:numel(cand)
        if isfield(S, cand{k})
            y = S.(cand{k});
            break;
        end
    end
    if isempty(y)
        error('No continuous target found in %s set. Expected one of: %s', whichSet, strjoin(cand, ', '));
    end
end

function [rmse, mae, r2, r] = computeRegMetrics(ytrue, ypred)
    ytrue = double(ytrue(:));
    ypred = double(ypred(:));
    dif = ypred - ytrue;

    rmse = sqrt(mean(dif.^2));
    mae  = mean(abs(dif));

    ybar = mean(ytrue);
    sst  = sum((ytrue - ybar).^2);
    sse  = sum((ytrue - ypred).^2);

    if sst <= eps
        r2 = NaN;
    else
        r2 = 1 - sse/sst;
    end

    if std(ytrue) <= eps || std(ypred) <= eps
        r = NaN;
    else
        cc = corrcoef(ytrue, ypred);
        r = cc(1,2);
    end
end

function [sliceStarts, sliceEnds] = computeSlicePartitions(DataForStat, indices)
    nSlices = numel(indices);
    sliceStarts = zeros(nSlices,1);
    sliceEnds   = zeros(nSlices,1);
    offset = 0;
    for s = 1:nSlices
        i = indices(s);
        nv = numel(DataForStat.row_pixCurFinal{i});
        sliceStarts(s) = offset + 1;
        sliceEnds(s)   = offset + nv;
        offset = offset + nv;
    end
end

function makeReconPanel(DataForStat, indices, sliceStarts, sliceEnds, valsAll, cl, outPath, panelTitle)
    nSlices = numel(indices);
    result = cell(1,nSlices);

    for s = 1:nSlices
        i = indices(s);
        mask = DataForStat.bwMaskLowFinal{i};
        img = nan(size(mask));

        rows = DataForStat.row_pixCurFinal{i};
        cols = DataForStat.col_pixCurFinal{i};
        vals = valsAll(sliceStarts(s):sliceEnds(s));

        linIdx = sub2ind(size(img), rows, cols);
        img(linIdx) = vals;
        result{s} = img;
    end

    fig = figure('Units','inches','Position',[1 1 16 5]);
    for s = 1:nSlices
        i = indices(s);
        subplot(1,nSlices,s);

        imagesc(result{s});
        colormap('jet');
        colorbar;
        clim(cl);
        axis off;

        lab = string(DataForStat.cases{i});
        if any(strcmpi(lab, ["CN","Cont","Control"]))
            tstr = "CN";
        elseif any(strcmpi(lab, ["AD","Alzheimer"]))
            tstr = "AD";
        else
            tstr = lab;
        end
        title(tstr, 'FontName','Arial');
    end
    sgtitle(panelTitle, 'FontName','Arial');

    set(fig, 'PaperPositionMode', 'auto');
    print(fig, outPath, '-dpng', '-r600');
    close(fig);

    fprintf('Saved: %s\n', outPath);
end
