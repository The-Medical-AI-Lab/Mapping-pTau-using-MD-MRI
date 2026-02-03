
clc; clear; close all;

%% 1. Load dataset and struct

load("dataset/regression_data_T2D.mat");  % X and y
load('dataset/DataForStat_101124.mat');  % DataForStat for reconstruction
rng(0);

%% 2. Compute slice boundaries for reconstruction

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

%% 3. Define slice-level group labels (CD/AD)

sliceLabels = zeros(nSlices,1);
for i = 1:nSlices
    sliceLabels(i) = strcmp(DataForStat.cases{i}, 'AD');
end

sliceVoxels = cell(nSlices,1);
for i = 1:nSlices
    sliceVoxels{i} = sliceStarts(i):sliceEnds(i);
end

cd_slices = find(sliceLabels == 0);
ad_slices = find(sliceLabels == 1);
k = min(numel(cd_slices), numel(ad_slices));
fprintf('Running %d-fold slice-wise CV with balanced CD-AD folds.\n', k);

%% 4. PCA (retain 95% variance)

[coeff, score, ~, ~, explained, mu] = pca(X);
cumExpl = cumsum(explained);
numComp = find(cumExpl >= 95, 1);
fprintf('Using %d principal components (%.2f%% variance)\n', numComp, cumExpl(numComp));
X_pca = score(:, 1:numComp);

%% 5. Initialize metric storage

mse = struct('lin', zeros(k,1), 'poly', zeros(k,1), 'svm_rbf', zeros(k,1), 'svm_lin', zeros(k,1), 'rf', zeros(k,1), 'mlp', zeros(k,1));
rsq = struct('lin', zeros(k,1), 'poly', zeros(k,1), 'svm_rbf', zeros(k,1), 'svm_lin', zeros(k,1), 'rf', zeros(k,1), 'mlp', zeros(k,1));
preds = struct('lin', nan(N,1), 'poly', nan(N,1), 'svm_rbf', nan(N,1), 'svm_lin', nan(N,1), 'rf', nan(N,1), 'mlp', nan(N,1));

%% 6. Balanced Slice-Wise Regression CV loop

for fold = 1:k
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

    % Linear
    mdl = fitlm(Xtr, ytr);
    ypred = predict(mdl, Xte);
    preds.lin(test_idx) = ypred;
    [mse.lin(fold), rsq.lin(fold)] = computeRegressionMetrics(yte, ypred);

    % Polynomial (deg 2)
    mdl = fitlm([Xtr, Xtr.^2], ytr);
    ypred = predict(mdl, [Xte, Xte.^2]);
    preds.poly(test_idx) = ypred;
    [mse.poly(fold), rsq.poly(fold)] = computeRegressionMetrics(yte, ypred);

    % SVM RBF
    mdl = fitrsvm(Xtr, ytr, 'KernelFunction','rbf', 'Standardize',true);
    ypred = predict(mdl, Xte);
    preds.svm_rbf(test_idx) = ypred;
    [mse.svm_rbf(fold), rsq.svm_rbf(fold)] = computeRegressionMetrics(yte, ypred);

    % SVM Linear
    mdl = fitrsvm(Xtr, ytr, 'KernelFunction','linear', 'Standardize',true);
    ypred = predict(mdl, Xte);
    preds.svm_lin(test_idx) = ypred;
    [mse.svm_lin(fold), rsq.svm_lin(fold)] = computeRegressionMetrics(yte, ypred);

    % Random Forest
    rf = TreeBagger(50, Xtr, ytr, 'Method','regression', 'OOBPrediction','off');
    ypred = predict(rf, Xte);
    preds.rf(test_idx) = ypred;
    [mse.rf(fold), rsq.rf(fold)] = computeRegressionMetrics(yte, ypred);

    % MLP
    mlp = fitrnet(Xtr, ytr, 'Standardize',true);
    ypred = predict(mlp, Xte);
    preds.mlp(test_idx) = ypred;
    [mse.mlp(fold), rsq.mlp(fold)] = computeRegressionMetrics(yte, ypred);
end

%% 7. Display CV results

fprintf('\n===== Slicewise Regression CV Performance =====\n');
models = {'Linear','Polynomial','SVM RBF','SVM Linear','Random Forest','MLP'};
fields = {'lin','poly','svm_rbf','svm_lin','rf','mlp'};
for i = 1:numel(models)
    f = fields{i};
    fprintf('%-14s: MSE = %.4f ± %.4f, R^2 = %.4f ± %.4f\n', models{i}, mean(mse.(f)), std(mse.(f)), mean(rsq.(f)), std(rsq.(f)));
end

%% 8. Plot all metrics

mse_mean = [mean(mse.lin), mean(mse.poly), mean(mse.svm_rbf), mean(mse.svm_lin), mean(mse.rf), mean(mse.mlp)];
rsq_mean = [mean(rsq.lin), mean(rsq.poly), mean(rsq.svm_rbf), mean(rsq.svm_lin), mean(rsq.rf), mean(rsq.mlp)];
mse_std = [std(mse.lin), std(mse.poly), std(mse.svm_rbf), std(mse.svm_lin), std(mse.rf), std(mse.mlp)];
rsq_std = [std(rsq.lin), std(rsq.poly), std(rsq.svm_rbf), std(rsq.svm_lin), std(rsq.rf), std(rsq.mlp)];

fig = figure;
set(fig, 'Position', [100, 100, 1000, 600]);

subplot(1,2,1);
bar(mse_mean);
hold on;
errorbar(1:6, mse_mean, mse_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', models);
ylabel('MSE');
title('Mean Squared Error');
grid on;

subplot(1,2,2);
bar(rsq_mean);
hold on;
errorbar(1:6, rsq_mean, rsq_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', models);
ylabel('R²');
title('R-Squared');
grid on;

sgtitle('5-Fold CV Regression Performance of T2D');
saveas(fig, 'image/slicewise regression/slicewise_performance_crossvalidation_T2D_pca.png');

%% 9. Reconstruct & plot regression maps

for m = 1:numel(fields)
    field = fields{m};
    result = cell(1,nSlices);
    for i = 1:nSlices
        mask = DataForStat.bwMaskLowFinal{i};
        img = zeros(size(mask));
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
        title(DataForStat.cases{i}, 'FontName','Arial');
    end
    sgtitle(sprintf('Reconstructed - %s Regression', models{m}), 'FontName','Arial');
    fname = sprintf('image/slicewise regression/slicewise_reconstruction_%s_T2D_pca.png', lower(models{m}));
    saveas(gcf, fname);
end

%% Helper

function [mse_val, rsq_val] = computeRegressionMetrics(yT, yP)
    mse_val = mean((yT - yP).^2);
    ss_res = sum((yT - yP).^2);
    ss_tot = sum((yT - mean(yT)).^2);
    rsq_val = 1 - ss_res / ss_tot;
end
