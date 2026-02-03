
clc; clear; close all;

%% 1. Load dataset and struct

load("dataset/classification_data_T2D_ternary.mat");  % X and yContinuous
y = yClassMean;  % Assume values in [0.15, 0.53, 0.91]
load('dataset/DataForStat_101124.mat');
rng(0);

%% 2. Map continuous target value to class labels (0,1,2)

% Extract unique values from target value
unique_vals = unique(y);
% disp(unique_vals);

% Map continuous target value to class labels 0,1,2
y_class = zeros(size(y));
y_class(y == unique_vals(1)) = 0;
y_class(y == unique_vals(2)) = 1;
y_class(y == unique_vals(3)) = 2;
y = y_class;

%% 3. Compute slice boundaries for reconstruction

nSlices = numel(DataForStat.bwMaskLowFinal);
sliceStarts = zeros(nSlices,1);
sliceEnds = zeros(nSlices,1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i) = offset + nv;
    offset = offset + nv;
end

N = size(X,1);

%% 4. Define slice-level group labels (CD/AD) and patientID

% Record slice group label (0 = CD, 1 = AD)
sliceLabels = zeros(nSlices,1);
for i = 1:nSlices
    sliceLabels(i) = strcmp(DataForStat.cases{i}, 'AD');
end

% Group voxel index by slice
sliceVoxels = cell(nSlices,1);
for i = 1:nSlices
    sliceVoxels{i} = sliceStarts(i):sliceEnds(i);
end

% Get slice indices for each group
cd_slices = find(sliceLabels == 0);
ad_slices = find(sliceLabels == 1);

% Number of crossval folds
k = min(numel(cd_slices), numel(ad_slices));
fprintf('Running %d-fold slice-wise CV with balanced CD-AD folds.\n', k);

%% 5. PCA (retain 95% variance)

[coeff, score, ~, ~, explained, mu] = pca(X);
cumExpl = cumsum(explained);
numComp = find(cumExpl >= 95, 1);
fprintf('Using %d principal components (%.2f%% variance)\n', numComp, cumExpl(numComp));
X_pca = score(:, 1:numComp);

%% 6. Initialize metric storage

models = {'SVM RBF', 'SVM Linear', 'Random Forest', 'MLP'};
acc = zeros(k, numel(models));
prec = zeros(k, numel(models));
rec = zeros(k, numel(models));
kappa = zeros(k, numel(models));
preds = struct('rbf',zeros(N,1), 'lin',zeros(N,1), 'rf',zeros(N,1), 'mlp',zeros(N,1));

%% 7. Balanced Slice-Wise Cross-Validation loop

for fold = 1:k
    fprintf('--- Fold %d/%d ---\n', fold, k);

    cd_idx = cd_slices(fold);
    ad_idx = ad_slices(fold);
    test_slices = [cd_idx, ad_idx];
    test_idx = [];
    for s = test_slices
        test_idx = [test_idx, sliceVoxels{s}];
    end
    train_idx = setdiff(1:N, test_idx);

    Xtr = X_pca(train_idx,:);
    ytr = y(train_idx);
    Xte = X_pca(test_idx,:);
    yte = y(test_idx);

    % SVM RBF
    fprintf('Training SVM RBF...\n');
    svmRBF = fitcecoc(Xtr, ytr, 'Learners', templateSVM('KernelFunction','rbf','Standardize',true));
    ypred = predict(svmRBF, Xte);
    preds.rbf(test_idx) = ypred;
    [acc(fold,1), prec(fold,1), rec(fold,1), kappa(fold,1)] = computeMetrics(yte, ypred);
    
    % SVM Linear
    fprintf('Training SVM Linear...\n');
    svmLin = fitcecoc(Xtr, ytr, 'Learners', templateSVM('KernelFunction','linear','Standardize',true));
    ypred = predict(svmLin, Xte);
    preds.lin(test_idx) = ypred;
    [acc(fold,2), prec(fold,2), rec(fold,2), kappa(fold,2)] = computeMetrics(yte, ypred);
    
    % Random Forest
    fprintf('Training Random Forest...\n');
    rf = TreeBagger(100, Xtr, ytr, 'Method','classification');
    ypred = str2double(predict(rf, Xte));
    preds.rf(test_idx)  = ypred;
    [acc(fold,3), prec(fold,3), rec(fold,3), kappa(fold,3)] = computeMetrics(yte, ypred);
    
    % MLP
    fprintf('Training MLP...\n');
    % --- Convert ytr (0/1/2) to one-hot matrix [3 x numSamples] ---
    ytr_onehot = full(ind2vec(ytr' + 1));  % ind2vec expects 1-based labels
    % --- Create MLP network ---
    mlp = patternnet(10);  % One hidden layer with 10 neurons
    mlp.layers{1}.transferFcn = 'poslin';  % Use ReLU activation
    mlp.trainParam.showWindow = false;     % Disable training GUI window
    % --- Train the network ---
    mlp = train(mlp, Xtr', ytr_onehot);
    % --- Predict class probabilities ---
    yprob = mlp(Xte');                  % Output is softmax probability matrix [3 x numTest]
    [~, ypred] = max(yprob, [], 1);     % Take the index of the max probability
    ypred = ypred' - 1;                 % Convert from 1/2/3 back to 0/1/2
    % --- Store predictions and compute metrics ---
    preds.mlp(test_idx) = ypred;
    [acc(fold,4), prec(fold,4), rec(fold,4), kappa(fold,4)] = computeMetrics(yte, ypred);
end

%% 8. Display results

fprintf('\n===== %d-Fold CV Performance =====\n', k);
fprintf('Model         Acc(μ±σ)     Prec(μ±σ)    Rec(μ±σ)     Kappa(μ±σ)\n');
for m = 1:numel(models)
    fprintf('%-13s %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', models{m}, mean(acc(:,m)), std(acc(:,m)), mean(prec(:,m)), std(prec(:,m)), mean(rec(:,m)), std(rec(:,m)), mean(kappa(:,m)), std(kappa(:,m)));
end

%% 9. Plot metrics

acc_mean = mean(acc);
prec_mean = mean(prec);
rec_mean = mean(rec);
kappa_mean = mean(kappa);

acc_std = std(acc);
prec_std = std(prec);
rec_std = std(rec);
kappa_std = std(kappa);

fig = figure;
set(fig, 'Position', [100, 100, 1600, 800]);

subplot(2,2,1);
bar(acc_mean);
hold on; errorbar(1:4, acc_mean, acc_std, '.k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', models);
ylabel('Accuracy');
ylim([0 1]);
title('Accuracy');
grid on;

subplot(2,2,2);
bar(prec_mean);
hold on; errorbar(1:4, prec_mean, prec_std, '.k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', models);
ylabel('Precision');
ylim([0 1]);
title('Precision');
grid on;

subplot(2,2,3);
bar(rec_mean);
hold on; errorbar(1:4, rec_mean, rec_std, '.k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', models);
ylabel('Recall');
ylim([0 1]);
title('Recall');
grid on;

subplot(2,2,4);
bar(kappa_mean);
hold on; errorbar(1:4, kappa_mean, kappa_std, '.k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', models);
ylabel('Cohen''s Kappa');
ylim([0 1]);
title('Cohen''s Kappa');
grid on;

sgtitle('Slicewise CV Performance Metrics for T2D (Ternary) under PCA');
saveas(fig, 'image/slicewise classification ternary/performance_slicewise_cv_T2D_ternary_pca.png');

%% 10. Reconstruct & plot classification maps

label_map = unique_vals;
fields = {'rbf','lin','rf','mlp'};

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
        title(DataForStat.cases{i}, 'FontName','Arial');
    end
    sgtitle(sprintf('Reconstructed - %s', strrep(models{m},'_',' ')),'FontName','Arial');
    fname = sprintf('image/slicewise classification ternary/reconstruction_%s_T2D_ternary_pca.png', lower(models{m}));
    saveas(gcf, fname);
end

%% Helper

function [ac, pr, re, ka] = computeMetrics(yT, yP)
    C = confusionmat(yT, yP);
    ac = sum(diag(C)) / sum(C(:));
    pr = mean(diag(C) ./ max(sum(C,1)',1));  % macro precision
    re = mean(diag(C) ./ max(sum(C,2),1));  % macro recall

    N = sum(C(:));
    Po = ac;
    Pe = sum(sum(C,1).*sum(C,2)) / N^2;
    ka = (Po - Pe) / (1 - Pe);
end
