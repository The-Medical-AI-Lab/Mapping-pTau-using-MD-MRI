
clear; clc; close all;

%% Paths

trainFile = "dataset/classification_data_T1D_binary.mat";
testFile = "external dataset/classification_data_T1D_binary_external.mat";
metaFile = "external dataset/DataForStat_120825.mat";
outDir = "image/external classification binary";
modelOutDir = "model/model_external";
resultMat = fullfile(modelOutDir, "external_results_all_models_T1D_binary.mat");

%% Load train and test

S_tr = load(trainFile);
S_te = load(testFile);

Xtr = S_tr.X;
Xte = S_te.X;

ytr = pickBinaryTarget(S_tr,'training');
yte = pickBinaryTarget(S_te,'external');

ytr = ytr(:);
yte = yte(:);

% ensure 0/1 numeric
ytr = double(ytr > 0);
yte = double(yte > 0);

fprintf('Loaded training: X=%s, y=%s\n', mat2str(size(Xtr)), mat2str(size(ytr)));
fprintf('Loaded external: X=%s, y=%s\n', mat2str(size(Xte)), mat2str(size(yte)));

%% PCA (>=95%%)

[coeff, ~, ~, ~, explained, mu] = pca(Xtr);
kPCA = find(cumsum(explained) >= 95, 1, 'first');
fprintf('PCA: using %d PCs (%.2f%% variance)\n', kPCA, sum(explained(1:kPCA)));

Xtr_pca = (Xtr - mu) * coeff(:, 1:kPCA);
Xte_pca = (Xte - mu) * coeff(:, 1:kPCA);

fprintf('Xtr original size: %d × %d\n', size(Xtr,1), size(Xtr,2));
fprintf('Xtr PCA size: %d × %d\n', size(Xtr_pca,1), size(Xtr_pca,2));
fprintf('Xte original size: %d × %d\n', size(Xte,1), size(Xte,2));
fprintf('Xte PCA size: %d × %d\n', size(Xte_pca,1), size(Xte_pca,2));

%% Model list (Binary Classification)

models = { ...
    struct('name','Logistic','field','logit'), ...
    struct('name','FLD','field','fld'), ...
    struct('name','SVM RBF','field','svm_rbf'), ...
    struct('name','SVM Linear','field','svm_lin'), ...
    struct('name','Random Forest','field','rf'), ...
    struct('name','MLP','field','mlp') ...
};

fields = cellfun(@(s) s.field, models, 'UniformOutput', false);
names = cellfun(@(s) s.name, models, 'UniformOutput', false);

% Store results struct
R = struct();
R.trainFile = trainFile;
R.testFile = testFile;
R.metaFile = metaFile;
R.kPCA = kPCA;
R.explained = explained;
R.mu = mu;
R.coeff = coeff;

%% Hyperparameter tuning on TRAIN only (5-fold CV)

innerCV = cvpartition(numel(ytr), 'KFold', 5);

for mi = 1:numel(models)
    name  = models{mi}.name;
    field = models{mi}.field;

    fprintf('\n==================== [%s] Tuning on TRAIN ====================\n', name);

    switch field

        %% Logistic regression
        case 'logit'

            opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_tuned = fitclinear( ...
                Xtr_pca, ytr, ...
                'Learner','logistic', ...
                'ClassNames',[0 1], ...
                'OptimizeHyperparameters', {'Lambda','Regularization'}, ...
                'HyperparameterOptimizationOptions', opts);

            best = mdl_tuned.ModelParameters;

            mdl_final = fitclinear( ...
                Xtr_pca, ytr, ...
                'Learner','logistic', ...
                'ClassNames',[0 1], ...
                'Lambda', best.Lambda, ...
                'Regularization', best.Regularization);
        
        %% FLD
        case 'fld'

            opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_tuned = fitcdiscr( ...
                Xtr_pca, ytr, ...
                'DiscrimType','linear', ...
                'ClassNames',[0 1], ...
                'OptimizeHyperparameters', {'Gamma','Delta'}, ...
                'HyperparameterOptimizationOptions', opts);

            best = mdl_tuned.HyperparameterOptimizationResults.XAtMinObjective;

            mdl_final = fitcdiscr( ...
                Xtr_pca, ytr, ...
                'DiscrimType', 'linear', ...
                'ClassNames',[0 1], ...
                'Gamma', best.Gamma, ...
                'Delta', best.Delta);

        %% SVM Linear
        case 'svm_lin'

            opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_tuned = fitcsvm( ...
                Xtr_pca, ytr, ...
                'KernelFunction','linear', ...
                'Standardize',true, ...
                'ClassNames',[0 1], ...
                'OptimizeHyperparameters', {'BoxConstraint'}, ...
                'HyperparameterOptimizationOptions', opts);

            best = mdl_tuned.ModelParameters;

            mdl_final = fitcsvm( ...
                Xtr_pca, ytr, ...
                'KernelFunction','linear', ...
                'Standardize',true, ...
                'ClassNames',[0 1], ...
                'BoxConstraint', best.BoxConstraint);

        %% SVM RBF
        case 'svm_rbf'

            opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_tuned = fitcsvm( ...
                Xtr_pca, ytr, ...
                'KernelFunction','rbf', ...
                'Standardize',true, ...
                'ClassNames',[0 1], ...
                'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
                'HyperparameterOptimizationOptions', opts);

            best = mdl_tuned.ModelParameters;

            mdl_final = fitcsvm( ...
                Xtr_pca, ytr, ...
                'KernelFunction','rbf', ...
                'Standardize',true, ...
                'ClassNames',[0 1], ...
                'BoxConstraint', best.BoxConstraint, ...
                'KernelScale', best.KernelScale);

        %% Random Forest
        case 'rf'
            opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_tuned = fitcensemble( ...
                Xtr_pca, ytr, ...
                'Method','Bag', ...
                'ClassNames',[0 1], ...
                'OptimizeHyperparameters', {'NumLearningCycles','MinLeafSize','MaxNumSplits'}, ...
                'HyperparameterOptimizationOptions', opts);

            best = mdl_tuned.HyperparameterOptimizationResults.XAtMinObjective;
            tTree = templateTree('MinLeafSize', best.MinLeafSize, 'MaxNumSplits', best.MaxNumSplits);

            mdl_final = fitcensemble( ...
                Xtr_pca, ytr, ...
                'Method','Bag', ...
                'ClassNames',[0 1], ...
                'Learners', tTree, ...
                'NumLearningCycles', round(best.NumLearningCycles));

        %% MLP
        case 'mlp'
            opts = struct('Optimizer','bayesopt', ...
                'AcquisitionFunctionName','expected-improvement-plus', ...
                'CVPartition',innerCV, ...
                'MaxObjectiveEvaluations',10, ...
                'ShowPlots',true, ...
                'Verbose',0);

            mdl_tuned = fitcnet( ...
                Xtr_pca, ytr, ...
                'Standardize',true, ...
                'IterationLimit',1000, ...
                'ClassNames',[0 1], ...
                'OptimizeHyperparameters', {'LayerSizes','Activations'}, ...
                'HyperparameterOptimizationOptions', opts);

            best = mdl_tuned.ModelParameters;

            mdl_final = fitcnet( ...
                Xtr_pca, ytr, ...
                'Standardize',true, ...
                'IterationLimit',1000, ...
                'ClassNames',[0 1], ...
                'LayerSizes', best.LayerSizes, ...
                'Activations', best.Activations);

        otherwise
            error('Unknown model field: %s', field);
    end

    %% Training metrics

    [yhat_tr, score_tr] = predictBinary(mdl_final, Xtr_pca);
    [acc_tr, prec_tr, rec_tr, f1_tr, auc_tr, kappa_tr] = computeClsMetrics(ytr, yhat_tr, score_tr);

    fprintf('[%s] Training (resub): Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f | AUC=%.4f | Kappa=%.4f\n', name, acc_tr, prec_tr, rec_tr, f1_tr, auc_tr, kappa_tr);

    R.(field).acc_tr = acc_tr;
    R.(field).prec_tr = prec_tr;
    R.(field).rec_tr = rec_tr;
    R.(field).f1_tr = f1_tr;
    R.(field).auc_tr = auc_tr;
    R.(field).kappa_tr = kappa_tr;

    %% External prediction

    fprintf('[%s] Predicting on External\n', name);

    [yhat, score] = predictBinary(mdl_final, Xte_pca);
    [acc, prec, rec, f1, auc, kappa] = computeClsMetrics(yte, yhat, score);

    fprintf('[%s] External: Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f | AUC=%.4f | Kappa=%.4f\n', name, acc, prec, rec, f1, auc, kappa);

    R.(field).name = name;
    R.(field).acc = acc;
    R.(field).prec = prec;
    R.(field).rec = rec;
    R.(field).f1 = f1;
    R.(field).auc = auc;
    R.(field).kappa  = kappa;
    R.(field).yhat = yhat;
    R.(field).score = score;

    %% Confusion matrix

    figC = figure('Units','inches','Position',[1 1 5.8 5.2]);
    cm = confusionmat(yte, yhat, 'Order', [0 1]);
    imagesc(cm);
    axis square;
    colormap('parula');
    colorbar;
    xticks([1 2]);
    yticks([1 2]);
    xticklabels({'0','1'});
    yticklabels({'0','1'});
    xlabel('Predicted');
    ylabel('True');
    title(sprintf('%s External Confusion Matrix', name), 'Interpreter','none');
    text(1,1,num2str(cm(1,1)),'HorizontalAlignment','center','Color','w','FontSize',12);
    text(2,1,num2str(cm(1,2)),'HorizontalAlignment','center','Color','w','FontSize',12);
    text(1,2,num2str(cm(2,1)),'HorizontalAlignment','center','Color','w','FontSize',12);
    text(2,2,num2str(cm(2,2)),'HorizontalAlignment','center','Color','w','FontSize',12);

    set(figC,'PaperPositionMode','auto');
    print(figC, fullfile(outDir, sprintf('cm_%s_external.png', field)), '-dpng','-r600');
    close(figC);

    %% ROC curve (external) if AUC valid

    if ~isnan(auc)
        figR = figure('Units','inches','Position',[1 1 6 5]);
        [fp,tp,~,AUC] = perfcurve(yte, score, 1);
        plot(fp,tp,'LineWidth',1.5); grid on; axis square;
        xlabel('False positive rate'); ylabel('True positive rate');
        title(sprintf('%s External ROC (AUC=%.3f)', name, AUC), 'Interpreter','none');
        set(figR,'PaperPositionMode','auto');
        print(figR, fullfile(outDir, sprintf('roc_%s_external.png', field)), '-dpng','-r600');
        close(figR);
    end
end

%% Summary bar plot (training classification metrics)

acc_tr_all = zeros(numel(fields),1);
prec_tr_all = zeros(numel(fields),1);
rec_tr_all = zeros(numel(fields),1);
kappa_tr_all = zeros(numel(fields),1);

for i = 1:numel(fields)
    f = fields{i};
    acc_tr_all(i) = R.(f).acc_tr;
    prec_tr_all(i) = R.(f).prec_tr;
    rec_tr_all(i) = R.(f).rec_tr;
    kappa_tr_all(i) = R.(f).kappa_tr;
end

figTr = figure('Units','inches','Position',[1 1 14 8]);

subplot(2,2,1);
bar(acc_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Accuracy');
title('Training Accuracy (Resubstitution)');
grid on;

subplot(2,2,2);
bar(prec_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Precision');
title('Training Precision (Resubstitution)');
grid on;

subplot(2,2,3);
bar(rec_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Recall');
title('Training Recall (Resubstitution)');
grid on;

subplot(2,2,4);
bar(kappa_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Cohen''s \kappa');
title('Training Cohen''s \kappa (Resubstitution)');
grid on;

sgtitle('Training Metrics (T1D Binary Classification)');

set(figTr,'PaperPositionMode','auto');
print(figTr, fullfile(outDir,'training_metrics_all_models.png'), '-dpng','-r600');
close(figTr);

%% Summary bar plot (external classification metrics)

acc_all = zeros(numel(fields),1);
prec_all = zeros(numel(fields),1);
rec_all = zeros(numel(fields),1);
kappa_all = zeros(numel(fields),1);

for i = 1:numel(fields)
    f = fields{i};
    acc_all(i) = R.(f).acc;
    prec_all(i) = R.(f).prec;
    rec_all(i) = R.(f).rec;
    kappa_all(i) = R.(f).kappa;
end

figM = figure('Units','inches','Position',[1 1 14 8]);

subplot(2,2,1);
bar(acc_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Accuracy');
title('External Accuracy');
grid on;

subplot(2,2,2);
bar(prec_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Precision');
title('External Precision');
grid on;

subplot(2,2,3);
bar(rec_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Recall');
title('External Recall');
grid on;

subplot(2,2,4);
bar(kappa_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Cohen''s \kappa');
title('External Cohen''s \kappa');
grid on;

sgtitle('External Test Metrics (T1D Binary Classification)');

set(figM,'PaperPositionMode','auto');
print(figM, fullfile(outDir,'external_metrics_all_models.png'), '-dpng','-r600');
close(figM);

%% Reconstruct maps: Predicted vs GroundTruth for each model

M = load(metaFile);
if isfield(M,'DataForStat2')
    DataForStat = M.DataForStat2;
else
    error('Expected DataForStat2 in %s', metaFile);
end

indices = 1:6;
[sliceStarts, sliceEnds] = computeSlicePartitions(DataForStat, indices);

assert(sliceEnds(end) == numel(yte), ...
    'Partition mismatch: sum slice voxels (%d) != length(yte) (%d). Check indices/order.', ...
    sliceEnds(end), numel(yte));

% binary display limits
cl = [0 1];

for i = 1:numel(fields)
    f = fields{i};

    makeReconPanel(DataForStat, indices, sliceStarts, sliceEnds, double(R.(f).yhat(:)), cl, ...
        fullfile(outDir, sprintf('recon_%s_pred_external.png', f)), ...
        sprintf('%s - Predicted (external)', R.(f).name));

    makeReconPanel(DataForStat, indices, sliceStarts, sliceEnds, double(yte(:)), cl, ...
        fullfile(outDir, sprintf('recon_%s_true_external.png', f)), ...
        sprintf('%s - GroundTruth (external)', R.(f).name));
end


%% Helper functions

function y = pickBinaryTarget(S, whichSet)
    % Try common binary label field names
    cand = {'yBin','yBinary','yCls','yClass','yLabel','y'};
    y = [];
    for k = 1:numel(cand)
        if isfield(S, cand{k})
            y = S.(cand{k});
            break;
        end
    end
    if isempty(y)
        error('No binary target found in %s set. Expected one of: %s', ...
            whichSet, strjoin(cand, ', '));
    end

    % Convert to numeric 0/1 if possible
    if islogical(y)
        y = double(y);
    elseif iscategorical(y)
        cats = categories(y);
        posCat = cats{end};
        y = double(y == posCat);
    elseif isstring(y) || ischar(y)
        error('Binary labels are string/char; please convert to 0/1 in the .mat first.');
    end
end

function [yhat, score1] = predictBinary(mdl, X)
    % Return predicted label (0/1) and a "score" for class 1 (probability if available)
    [pred, score] = predict(mdl, X);

    % pred might be categorical, logical, or numeric
    if iscategorical(pred)
        cats = categories(pred);
        pred = double(pred == cats{end});
    else
        pred = double(pred > 0);
    end

    % score handling: try to extract class-1 column
    if isempty(score)
        score1 = double(pred); % fallback
        yhat  = pred;
        return;
    end

    % score can be N×2 for class [0 1]
    if size(score,2) == 2
        score1 = score(:,2);
    else
        score1 = score(:,1);
    end

    yhat = pred;
end

function [acc, prec, rec, f1, auc, kappa] = computeClsMetrics(ytrue, yhat, score1)
    ytrue = double(ytrue(:)>0);
    yhat  = double(yhat(:)>0);

    tp = sum((ytrue==1) & (yhat==1));
    tn = sum((ytrue==0) & (yhat==0));
    fp = sum((ytrue==0) & (yhat==1));
    fn = sum((ytrue==1) & (yhat==0));

    acc = (tp+tn) / max(1,(tp+tn+fp+fn));
    prec = tp / max(1,(tp+fp));
    rec  = tp / max(1,(tp+fn));
    f1   = 2*prec*rec / max(eps,(prec+rec));

    % Cohen's Kappa
    N = tp + tn + fp + fn;
    Po = (tp + tn) / max(1, N);
    Pe = ((tp+fp)*(tp+fn) + (fn+tn)*(fp+tn)) / max(1, N^2);
    kappa = (Po - Pe) / max(eps, (1 - Pe));

    % AUC
    auc = NaN;
    try
        if nargin >= 3 && ~isempty(score1) && numel(unique(ytrue))==2
            [~,~,~,auc] = perfcurve(ytrue, double(score1(:)), 1);
        end
    catch
        auc = NaN;
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
        colormap('jet'); colorbar;
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
