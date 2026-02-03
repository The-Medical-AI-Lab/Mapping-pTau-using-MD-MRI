
clc; clear; close all;

%% 1. Load dataset and struct

load("dataset/classification_data_T2D_binary.mat");  % X and yBinary
y = yBinary;
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

%% 3. Define slice-level group labels (CD/AD) and patientID

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

%% 4. PCA (retain 95% variance)

[coeff, score, ~, ~, explained, mu] = pca(X);
cumExpl = cumsum(explained);
numComp = find(cumExpl >= 95, 1);
fprintf('Using %d principal components (%.2f%% variance)\n', numComp, cumExpl(numComp));
X_pca = score(:, 1:numComp);

%% 5. Initialize metric storage

acc   = struct('log', zeros(k,1), 'fld', zeros(k,1), 'rbf', zeros(k,1), 'lin', zeros(k,1), 'rf', zeros(k,1), 'mlp', zeros(k,1));
prec  = struct('log', zeros(k,1), 'fld', zeros(k,1), 'rbf', zeros(k,1), 'lin', zeros(k,1), 'rf', zeros(k,1), 'mlp', zeros(k,1));
rec   = struct('log', zeros(k,1), 'fld', zeros(k,1), 'rbf', zeros(k,1), 'lin', zeros(k,1), 'rf', zeros(k,1), 'mlp', zeros(k,1));
kappa = struct('log', zeros(k,1), 'fld', zeros(k,1), 'rbf', zeros(k,1), 'lin', zeros(k,1), 'rf', zeros(k,1), 'mlp', zeros(k,1));
preds = struct('log', nan(N,1), 'fld', nan(N,1), 'rbf', nan(N,1), 'lin', nan(N,1), 'rf', nan(N,1), 'mlp', nan(N,1));

%% 6. Balanced Slice-Wise Cross-Validation loop

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

    % Logistic Regression
    mdl = fitglm(Xtr, ytr, 'Distribution','binomial','Link','logit');
    ypred = predict(mdl, Xte) >= 0.5;
    preds.log(test_idx) = ypred;
    [acc.log(fold), prec.log(fold), rec.log(fold), kappa.log(fold)] = computeMetrics(yte, ypred);

    % Fisher Linear Discriminant
    X1 = Xtr(ytr==1,:); X0 = Xtr(ytr==0,:);
    m1 = mean(X1)'; m0 = mean(X0)'; Sw = cov(X1)+cov(X0);
    w = Sw \ (m1-m0); thr = 0.5*(m1'*w + m0'*w);
    ypred = (Xte*w) > thr;
    preds.fld(test_idx) = ypred;
    [acc.fld(fold), prec.fld(fold), rec.fld(fold), kappa.fld(fold)] = computeMetrics(yte, ypred);

    % SVM RBF
    svmRBF = fitcsvm(Xtr, ytr, 'KernelFunction','rbf','KernelScale','auto','Standardize',true);
    ypred = predict(svmRBF, Xte);
    preds.rbf(test_idx) = ypred;
    [acc.rbf(fold), prec.rbf(fold), rec.rbf(fold), kappa.rbf(fold)] = computeMetrics(yte, ypred);

    % SVM Linear
    svmLin = fitcsvm(Xtr, ytr, 'KernelFunction','linear','Standardize',true);
    ypred = predict(svmLin, Xte);
    preds.lin(test_idx) = ypred;
    [acc.lin(fold), prec.lin(fold), rec.lin(fold), kappa.lin(fold)] = computeMetrics(yte, ypred);
    
    % Random Forest
    rf = TreeBagger(50, Xtr, ytr, 'Method','classification', 'OOBPrediction','off');
    ypred = predict(rf, Xte);
    ypred = str2double(ypred);
    ypred = cast(ypred, 'like', yte);
    preds.rf(test_idx) = ypred;
    [acc.rf(fold), prec.rf(fold), rec.rf(fold), kappa.rf(fold)] = computeMetrics(yte, ypred);

    % MLP
    mlp = fitcnet(Xtr, ytr, 'Standardize',true);
    ypred = predict(mlp, Xte);
    preds.mlp(test_idx) = ypred;
    [acc.mlp(fold), prec.mlp(fold), rec.mlp(fold), kappa.mlp(fold)] = computeMetrics(yte, ypred);
end

%% 7. Display CV results

fprintf('\n===== Slicewise CV Performance =====\n');
fprintf('Model           Acc(μ±σ)     Prec(μ±σ)    Rec(μ±σ)     Kappa(μ±σ)\n');
fprintf('Logistic      : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', mean(acc.log), std(acc.log), mean(prec.log), std(prec.log), mean(rec.log), std(rec.log), mean(kappa.log), std(kappa.log));
fprintf('FLD           : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', mean(acc.fld), std(acc.fld), mean(prec.fld), std(prec.fld), mean(rec.fld), std(rec.fld), mean(kappa.fld), std(kappa.fld));
fprintf('SVM (RBF)     : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', mean(acc.rbf), std(acc.rbf), mean(prec.rbf), std(prec.rbf), mean(rec.rbf), std(rec.rbf), mean(kappa.rbf), std(kappa.rbf));
fprintf('SVM (Linear)  : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', mean(acc.lin), std(acc.lin), mean(prec.lin), std(prec.lin), mean(rec.lin), std(rec.lin), mean(kappa.lin), std(kappa.lin));
fprintf('Random Forest : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', mean(acc.rf), std(acc.rf), mean(prec.rf), std(prec.rf), mean(rec.rf), std(rec.rf), mean(kappa.rf), std(kappa.rf));
fprintf('MLP           : %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f\n', mean(acc.mlp), std(acc.mlp), mean(prec.mlp), std(prec.mlp), mean(rec.mlp), std(rec.mlp), mean(kappa.mlp), std(kappa.mlp));

%% 8. Plot all metrics

acc_mean = [mean(acc.log), mean(acc.fld), mean(acc.rbf), mean(acc.lin), mean(acc.rf), mean(acc.mlp)];
prec_mean = [mean(prec.log), mean(prec.fld), mean(prec.rbf), mean(prec.lin), mean(prec.rf), mean(prec.mlp)];
rec_mean = [mean(rec.log), mean(rec.fld), mean(rec.rbf), mean(rec.lin), mean(rec.rf), mean(rec.mlp)];
kappa_mean = [mean(kappa.log), mean(kappa.fld), mean(kappa.rbf), mean(kappa.lin), mean(kappa.rf), mean(kappa.mlp)];

acc_std = [std(acc.log), std(acc.fld), std(acc.rbf), std(acc.lin), std(acc.rf), std(acc.mlp)];
prec_std = [std(prec.log), std(prec.fld), std(prec.rbf), std(prec.lin), std(prec.rf), std(prec.mlp)];
rec_std = [std(rec.log), std(prec.fld), std(prec.rbf), std(prec.lin), std(prec.rf), std(prec.mlp)];
kappa_std = [std(kappa.log), std(kappa.fld), std(kappa.rbf), std(kappa.lin), std(kappa.rf), std(kappa.mlp)];

model_labels = {'Logistic', 'FLD', 'SVM RBF', 'SVM Linear', 'RF', 'MLP'};

fig = figure;
set(fig, 'Position', [100, 100, 1300, 800]);

subplot(2,2,1);
bar(acc_mean); hold on;
errorbar(1:numel(acc_mean), acc_mean, acc_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Accuracy'); 
ylim([0 1]); 
title('Accuracy'); 
grid on;

subplot(2,2,2);
bar(prec_mean); hold on;
errorbar(1:numel(acc_mean), prec_mean, prec_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Precision'); 
ylim([0 1]); 
title('Precision'); 
grid on;

subplot(2,2,3);
bar(rec_mean); hold on;
errorbar(1:numel(acc_mean), rec_mean, rec_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Recall'); 
ylim([0 1]); 
title('Recall'); 
grid on;

subplot(2,2,4);
bar(kappa_mean); hold on;
errorbar(1:numel(acc_mean), kappa_mean, kappa_std, '.k', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', model_labels);
ylabel('Cohen''s Kappa'); 
ylim([0 1]); 
title('Cohen''s Kappa'); 
grid on;

sgtitle('Slicewise CV Performance Metrics for T2D under PCA');
saveas(fig, 'image/slicewise classification binary/performance_slice_cv_T2D_pca.png');

%% 9. Reconstruct & plot classification maps

models = {'Logistic','FLD','SVM_RBF','SVM_Linear','Random_Forest','MLP'};
fields = {'log','fld','rbf','lin','rf','mlp'};

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
    sgtitle(sprintf('Reconstructed - %s', strrep(models{m},'_',' ')),'FontName','Arial');
    fname = sprintf('image/slicewise classification binary/reconstruction_%s_T2D_pca.png', lower(models{m}));
    saveas(gcf, fname);
end

%% Helper

function [ac, pr, re, ka] = computeMetrics(yT, yP)
    C = confusionmat(yT, yP);
    TN = C(1,1); 
    FP = C(1,2); 
    FN = C(2,1); 
    TP = C(2,2);
    ac = (TP+TN)/sum(C(:));
    pr = TP/(TP+FP);
    re = TP/(TP+FN);
    Po = (TP+TN)/sum(C(:));
    Pe = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(sum(C(:))^2);
    ka = (Po-Pe)/(1-Pe);
end


