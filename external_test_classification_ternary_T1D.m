
clear; clc; close all;

%% Paths

trainFile = "dataset/classification_data_T1D_ternary.mat";
testFile = "external dataset/classification_data_T1D_ternary_external.mat";
metaFile = "external dataset/DataForStat_120825.mat";
outDir = "image/external classification ternary";
modelOutDir = "model/model_external";
resultMat = fullfile(modelOutDir, "external_results_all_models_T1D_ternary.mat");

%% Load train and test

S_tr = load(trainFile);
S_te = load(testFile);

Xtr = S_tr.X;
Xte = S_te.X;

ytr = pickTernaryTarget(S_tr,'training');
yte = pickTernaryTarget(S_te,'external');

ytr = ytr(:);
yte = yte(:);

u_tr = unique(ytr);
u_te = unique(yte);

ytr_class = zeros(size(ytr));
ytr_class(ytr == u_tr(1)) = 0;
ytr_class(ytr == u_tr(2)) = 1;
ytr_class(ytr == u_tr(3)) = 2;

yte_class = zeros(size(yte));
yte_class(yte == u_te(1)) = 0;
yte_class(yte == u_te(2)) = 1;
yte_class(yte == u_te(3)) = 2;

ytr = ytr_class;
yte = yte_class;

ytr = double(ytr);
yte = double(yte);

classNames = [0 1 2];

fprintf('Loaded training: X=%s, y=%s\n', mat2str(size(Xtr)), mat2str(size(ytr)));
fprintf('Loaded external: X=%s, y=%s\n', mat2str(size(Xte)), mat2str(size(yte)));
fprintf('Class counts (train): %s\n', mat2str(histcounts(ytr, [-0.5 0.5 1.5 2.5])));
fprintf('Class counts (ext): %s\n', mat2str(histcounts(yte, [-0.5 0.5 1.5 2.5])));

%% PCA (>=95%)

[coeff, ~, ~, ~, explained, mu] = pca(Xtr);
kPCA = find(cumsum(explained) >= 95, 1, 'first');
fprintf('PCA: using %d PCs (%.2f%% variance)\n', kPCA, sum(explained(1:kPCA)));

Xtr_pca = (Xtr - mu) * coeff(:, 1:kPCA);
Xte_pca = (Xte - mu) * coeff(:, 1:kPCA);

fprintf('Xtr original size: %d × %d\n', size(Xtr,1), size(Xtr,2));
fprintf('Xtr PCA size: %d × %d\n', size(Xtr_pca,1), size(Xtr_pca,2));
fprintf('Xte original size: %d × %d\n', size(Xte,1), size(Xte,2));
fprintf('Xte PCA size: %d × %d\n', size(Xte_pca,1), size(Xte_pca,2));

%% Model list (Ternary Classification)

models = { ...
    struct('name','Logistic','field','logit'), ...
    struct('name','FLD','field','fld'), ...
    struct('name','SVM RBF','field','svm_rbf'), ...
    struct('name','SVM Linear','field','svm_lin'), ...
    struct('name','Random Forest','field','rf'), ...
    struct('name','MLP','field','mlp') ...
};

fields = cellfun(@(s) s.field, models, 'UniformOutput', false);
names = cellfun(@(s) s.name,  models, 'UniformOutput', false);

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

innerCV = cvpartition(numel(ytr), 'KFold', 5);

for mi = 1:numel(models)
    name  = models{mi}.name;
    field = models{mi}.field;

    fprintf('\n==================== [%s] Tuning on TRAIN ====================\n', name);
    
    opts = struct( ...
        'Optimizer','bayesopt', ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'CVPartition', innerCV, ...
        'MaxObjectiveEvaluations', 10, ...
        'ShowPlots', true, ...
        'Verbose', 0);

    switch field
        %% Logistic
        case 'logit'

            % templateLinear with logistic learner; Lambda/Regularization
            tLogit = templateLinear('Learner','logistic', ...
                'Lambda',1e-4, ...
                'Regularization','ridge');
            
            mdl_final = fitcecoc( ...
                Xtr_pca, ytr, ...
                'Learners', tLogit, ...
                'Coding', 'onevsone', ...
                'ClassNames', [0 1 2], ...
                'OptimizeHyperparameters', {'Lambda','Regularization'}, ...
                'HyperparameterOptimizationOptions', opts);

        %% FLD
        case 'fld'

            mdl_final = fitcdiscr( ...
                Xtr_pca, ytr, ...
                'ClassNames', [0 1 2], ...
                'OptimizeHyperparameters', {'DiscrimType','Gamma','Delta'}, ...
                'HyperparameterOptimizationOptions', opts);

        %% SVM Linear
        case 'svm_lin'

            tSVMrbf = templateSVM('KernelFunction','rbf', ...
                'Standardize',true);
        
            mdl_final = fitcecoc( ...
                Xtr_pca, ytr, ...
                'Learners', tSVMrbf, ...
                'Coding', 'onevsone', ...
                'ClassNames', [0 1 2], ...
                'OptimizeHyperparameters', {'BoxConstraint','KernelScale'}, ...
                'HyperparameterOptimizationOptions', opts);

        %% SVM RBF
        case 'svm_rbf'

            tLinear = templateLinear('Learner','svm', ...
                'Lambda',1e-4, ...
                'Regularization','ridge');
            
            mdl_final = fitcecoc( ...
                Xtr_pca, ytr, ...
                'Learners', tLinear, ...
                'Coding','onevsone', ...
                'ClassNames',[0 1 2], ...
                'OptimizeHyperparameters', {'Lambda','Regularization'}, ...
                'HyperparameterOptimizationOptions', opts);

        %% Random Forest
        case 'rf'

            mdl_final = fitcensemble( ...
                Xtr_pca, ytr, ...
                'Method','Bag', ...
                'Learners','Tree', ...
                'ClassNames', [0 1 2], ...
                'OptimizeHyperparameters', {'NumLearningCycles','MinLeafSize','MaxNumSplits'}, ...
                'HyperparameterOptimizationOptions', opts);

        %% MLP
        case 'mlp'

            mdl_final = fitcnet( ...
                Xtr_pca, ytr, ...
                'ClassNames', [0 1 2], ...
                'Standardize', true, ...
                'IterationLimit', 500, ...
                'OptimizeHyperparameters', {'NumLayers','Activations','Layer_1_Size','Layer_2_Size','Layer_3_Size'}, ...
                'HyperparameterOptimizationOptions', opts);

        otherwise
            error('Unknown model field: %s', field);
    end

    %% TRAIN (resubstitution) metrics

    [yhat_tr, score_tr] = predictMulti(mdl_final, Xtr_pca, classNames);
    [acc_tr, balacc_tr, recmac_tr, precmac_tr, kappa_tr, f1macro_tr] = computeClsMetricsMulti(ytr, yhat_tr, classNames);

    fprintf('[%s] Training (resub): Acc=%.4f | BalAcc=%.4f | MacroRecall=%.4f | MacroPrecision=%.4f | MacroF1=%.4f | Kappa=%.4f\n', ...
        name, acc_tr, balacc_tr, recmac_tr, precmac_tr, f1macro_tr, kappa_tr);

    R.(field).acc_tr = acc_tr;
    R.(field).balacc_tr = balacc_tr;
    R.(field).precmac_tr = precmac_tr;
    R.(field).recmac_tr = recmac_tr;
    R.(field).f1mac_tr = f1macro_tr;
    R.(field).kappa_tr = kappa_tr;

    %% EXTERNAL prediction

    fprintf('[%s] Predicting on External\n', name);

    [yhat, score] = predictMulti(mdl_final, Xte_pca, classNames);
    [acc, balacc, recmacro, precmacro, kappa, f1macro] = computeClsMetricsMulti(yte, yhat, classNames);

    fprintf('[%s] External: Acc=%.4f | BalAcc=%.4f | MacroRecall=%.4f | MacroPrecision=%.4f | MacroF1=%.4f | Kappa=%.4f\n', ...
        name, acc, balacc, recmacro, precmacro, f1macro, kappa);

    R.(field).name = name;
    R.(field).acc = acc;
    R.(field).balacc = balacc;
    R.(field).precmac = precmacro;
    R.(field).recmac = recmacro;
    R.(field).f1mac = f1macro;
    R.(field).kappa = kappa;
    R.(field).yhat = yhat;
    R.(field).score = score;

    %% Confusion matrix (external)

    figC = figure('Units','inches','Position',[1 1 6.5 5.8]);
    cm = confusionmat(yte, yhat, 'Order', classNames);
    imagesc(cm);
    axis square;
    colormap('parula');
    colorbar;
    xticks(1:numel(classNames));
    yticks(1:numel(classNames));
    xticklabels(string(classNames));
    yticklabels(string(classNames));
    xlabel('Predicted');
    ylabel('True');
    title(sprintf('%s External Confusion Matrix', name), 'Interpreter','none');

    for rr = 1:size(cm,1)
        for cc = 1:size(cm,2)
            text(cc, rr, num2str(cm(rr,cc)), 'HorizontalAlignment','center', 'Color','w', 'FontSize',11);
        end
    end

    set(figC,'PaperPositionMode','auto');
    print(figC, fullfile(outDir, sprintf('cm_%s_external.png', field)), '-dpng','-r600');
    close(figC);

end

%% TRAINING SUMMARY PLOT (Accuracy / Macro Recall / Macro Precision / Kappa)

acc_tr_all = zeros(numel(fields),1);
recmac_tr_all = zeros(numel(fields),1);
precmac_tr_all = zeros(numel(fields),1);
kappa_tr_all = zeros(numel(fields),1);

for i = 1:numel(fields)
    f = fields{i};
    acc_tr_all(i) = R.(f).acc_tr;
    recmac_tr_all(i) = R.(f).recmac_tr;
    precmac_tr_all(i) = R.(f).precmac_tr;
    kappa_tr_all(i) = R.(f).kappa_tr;
end

figTr = figure('Units','inches','Position',[1 1 14 8]);

subplot(2,2,1);
bar(acc_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Accuracy');
title('Training Accuracy');
grid on;

subplot(2,2,2);
bar(recmac_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Macro Recall');
title('Training Macro Recall');
grid on;

subplot(2,2,3);
bar(precmac_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Macro Precision');
title('Training Macro Precision');
grid on;

subplot(2,2,4);
bar(kappa_tr_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Cohen''s \kappa');
title('Training Cohen''s \kappa');
grid on;

sgtitle('Training Metrics (T1D Ternary Classification)');
set(figTr,'PaperPositionMode','auto');
print(figTr, fullfile(outDir,'training_metrics_all_models.png'), '-dpng','-r600');
close(figTr);

%% EXTERNAL SUMMARY PLOT (Accuracy / Macro Recall / Macro Precision / Kappa)

acc_all = zeros(numel(fields),1);
recmac_all = zeros(numel(fields),1);
precmac_all = zeros(numel(fields),1);
kappa_all = zeros(numel(fields),1);

for i = 1:numel(fields)
    f = fields{i};
    acc_all(i) = R.(f).acc;
    recmac_all(i) = R.(f).recmac;
    precmac_all(i) = R.(f).precmac;
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
bar(recmac_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Macro Recall');
title('External Macro Recall');
grid on;

subplot(2,2,3);
bar(precmac_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Macro Precision');
title('External Macro Precision');
grid on;

subplot(2,2,4);
bar(kappa_all);
set(gca,'XTickLabel',names);
xtickangle(30);
ylabel('Cohen''s \kappa');
title('External Cohen''s \kappa');
grid on;

sgtitle('External Test Metrics (T1D Ternary Classification)');
set(figM,'PaperPositionMode','auto');
print(figM, fullfile(outDir,'external_metrics_all_models.png'), '-dpng','-r600');
close(figM);

%% RECONSTRUCTION (PRED LABELS)

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

cl = [min(classNames) max(classNames)];

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

function y = pickTernaryTarget(S, whichSet)
    % Try common ternary label field names
    cand = {'yTer','yTernary','yCls','yClass','yLabel','y','yClassMean'};
    y = [];
    for k = 1:numel(cand)
        if isfield(S, cand{k})
            y = S.(cand{k});
            break;
        end
    end
    if isempty(y)
        error('No ternary target found in %s set. Expected one of: %s', ...
            whichSet, strjoin(cand, ', '));
    end

    if islogical(y)
        y = double(y);
    elseif iscategorical(y)
        % map categories to 0..K-1
        cats = categories(y);
        ynum = zeros(numel(y),1);
        for i = 1:numel(cats)
            ynum(y == cats{i}) = i-1;
        end
        y = ynum;
    end

    y = double(y(:));

    u = unique(y);
    if numel(u) ~= 3
        warning('Expected 3 unique classes, but got %d: %s', numel(u), mat2str(u'));
    end
end

function [yhat, score] = predictMulti(mdl, X, classNames)
    % Return predicted labels and scores (N×K when available)
    [pred, sc] = predict(mdl, X);

    if iscategorical(pred)
        pred = double(grp2idx(pred)) - 1; % 0..K-1
    else
        pred = double(pred);
    end

    % some models return labels not in 0..K-1 -> map to classNames order
    if ~all(ismember(unique(pred), classNames))
        % try map by unique order
        u = unique(pred);
        map = containers.Map(num2cell(u), num2cell(0:numel(u)-1));
        for i = 1:numel(pred)
            if isKey(map, pred(i)), pred(i) = map(pred(i)); end
        end
    end

    yhat = pred;

    if nargin < 3, classNames = unique(yhat)'; end %#ok<NASGU>
    if isempty(sc)
        score = [];
    else
        score = sc; % typically N×K
    end
end

function [acc, balacc, recmacro, precmacro, kappa, f1macro] = computeClsMetricsMulti(ytrue, yhat, classNames)
    ytrue = double(ytrue(:));
    yhat  = double(yhat(:));

    % confusion matrix with fixed order
    C = confusionmat(ytrue, yhat, 'Order', classNames);
    N = sum(C(:));
    acc = trace(C) / max(1, N);

    % balanced accuracy = mean recall over classes
    recalls = zeros(numel(classNames),1);
    for i = 1:numel(classNames)
        tp = C(i,i);
        fn = sum(C(i,:)) - tp;
        recalls(i) = tp / max(eps, tp+fn);
    end
    balacc = mean(recalls);
    
    % Macro Precision and Macro Recall
    K = numel(classNames);
    precisions = zeros(K,1);
    recalls = zeros(K,1);
    
    for i = 1:K
        tp = C(i,i);
        fp = sum(C(:,i)) - tp;
        fn = sum(C(i,:)) - tp;
        precisions(i) = tp / max(eps, tp+fp);
        recalls(i) = tp / max(eps, tp+fn);
    end
    precmacro = mean(precisions);
    recmacro  = mean(recalls);

    % Macro F1
    f1s = zeros(numel(classNames),1);
    for i = 1:numel(classNames)
        tp = C(i,i);
        fp = sum(C(:,i)) - tp;
        fn = sum(C(i,:)) - tp;
        prec = tp / max(eps, tp+fp);
        rec  = tp / max(eps, tp+fn);
        f1s(i) = 2*prec*rec / max(eps, prec+rec);
    end
    f1macro = mean(f1s);

    % Cohen's Kappa (multiclass)
    Po = acc;
    rowSums = sum(C,2);
    colSums = sum(C,1);
    Pe = sum(rowSums .* colSums') / max(eps, N^2);
    kappa = (Po - Pe) / max(eps, (1 - Pe));
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
        title(lab, 'FontName','Arial');
    end
    sgtitle(panelTitle, 'FontName','Arial');

    set(fig, 'PaperPositionMode', 'auto');
    print(fig, outPath, '-dpng', '-r600');
    close(fig);

    fprintf('Saved: %s\n', outPath);
end
