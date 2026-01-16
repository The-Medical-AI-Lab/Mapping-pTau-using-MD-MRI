
clc; clear; close all;

%% 1) Load dataset and struct

load("dataset/regression_image_data_T1D.mat", 'X', 'y'); % X: 50x50x1xN images, y: Nx1 regression targets
load('dataset/DataForStat_101124.mat'); % For voxel-wise reconstruction
rng(0);

%% 2) Compute slice boundaries for reconstruction (voxel indexing per slice)
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

%% 3) 5-Fold Cross-Validation with inner Bayesian Optimization (BO)

N = size(X,4);
k = 5;
cv = cvpartition(N, 'KFold', k);

mse_cnn  = zeros(k,1);
r2_cnn   = zeros(k,1);
preds_cnn = zeros(N,1);

for fold = 1:k
    fprintf('\n=== Fold %d/%d ===\n', fold, k);

    % ---- Outer split: train/test ----
    trIdx = training(cv, fold);
    teIdx = test(cv, fold);

    XTrainAll = X(:,:,:,trIdx);
    YTrainAll = y(trIdx);
    XTest     = X(:,:,:,teIdx);
    YTest     = y(teIdx);

    % ---- Inner split: validation for BO (avoid leakage) ----
    inner = cvpartition(numel(YTrainAll), 'HoldOut', 0.2);
    XTr  = XTrainAll(:,:,:,training(inner));
    YTr  = YTrainAll(training(inner));
    XVal = XTrainAll(:,:,:,test(inner));
    YVal = YTrainAll(test(inner));

    % ---- Search space for BO ----
    vars = [
        optimizableVariable('initLR', [1e-4, 1e-2], 'Transform','log')
        optimizableVariable('l2', [1e-6, 1e-3], 'Transform','log')
        optimizableVariable('mb', [32, 128], 'Type','integer')
        optimizableVariable('opt', {'adam','sgdm','rmsprop'})
        optimizableVariable('drop', [0.0, 0.5])
        optimizableVariable('fc', [32, 128], 'Type','integer')
    ];

    % ---- Objective function: returns validation MSE (minimize) ----
    objFcn = @(T) cnnRegObjFcn(T, XTr, YTr, XVal, YVal);

    % ---- Run Bayesian Optimization ----
    results = bayesopt(objFcn, vars, ...
        'MaxObjectiveEvaluations', 18, ...
        'IsObjectiveDeterministic', false, ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'UseParallel', false, ... % set true if you have Parallel Toolbox + parpool
        'Verbose', 1);

    best = results.XAtMinEstimatedObjective;
    fprintf('Best params (fold %d): LR=%.2e, L2=%.2e, MB=%d, Opt=%s, Drop=%.2f, FC=%d\n', fold, best.initLR, best.l2, best.mb, string(best.opt), best.drop, best.fc);

    % ---- Re-train on the full outer training set with best hyperparams ----
    layers = buildLayers(best.fc, best.drop);
    finalOpts = trainingOptions(string(best.opt), ...
        'InitialLearnRate', best.initLR, ...
        'L2Regularization', best.l2, ...
        'MiniBatchSize', best.mb, ...
        'MaxEpochs', 25, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XVal, YVal}, ... % keep for early stopping signal
        'ValidationFrequency', 50, ...
        'ValidationPatience', 5, ...
        'ExecutionEnvironment','auto', ...
        'Verbose', true);

    net = trainNetwork(XTrainAll, YTrainAll, layers, finalOpts);

    % ---- Test evaluation ----
    YPred = predict(net, XTest);
    YPred = squeeze(YPred); % ensure column vector
    YTest = squeeze(YTest); % ensure column vector

    preds_cnn(teIdx) = YPred;

    mse_cnn(fold) = mean((YPred - YTest).^2);
    SSres = sum((YTest - YPred).^2);
    SStot = sum((YTest - mean(YTest)).^2);
    r2_cnn(fold) = 1 - SSres/SStot;

    fprintf('Fold %d - MSE: %.4f, R^2: %.4f\n', fold, mse_cnn(fold), r2_cnn(fold));
end

%% 4) Display average metrics
fprintf('\n===== CNN Regression (5-Fold CV with BO) =====\n');
fprintf('MSE: %.4f ± %.4f\n', mean(mse_cnn), std(mse_cnn));
fprintf('R^2: %.4f ± %.4f\n', mean(r2_cnn), std(r2_cnn));

%% 5) Plot metrics with error bars

figure;
% MSE
subplot(1,2,1);
bar(mean(mse_cnn)); hold on;
errorbar(1, mean(mse_cnn), std(mse_cnn), '.k', 'LineWidth', 1.5);
set(gca, 'XTick', 1, 'XTickLabel', {'MSE'});
ylabel('Score'); title('CNN Regression - MSE (BO)'); grid on;
% R^2
subplot(1,2,2);
bar(mean(r2_cnn)); hold on;
errorbar(1, mean(r2_cnn), std(r2_cnn), '.k', 'LineWidth', 1.5);
set(gca, 'XTick', 1, 'XTickLabel', {'R^2'});
ylabel('Score'); title('CNN Regression - R^2 (BO)'); grid on;
saveas(gcf, 'image/cnn regression/performance_crossvalidation_T1D_BO.png');

%% 6) Reconstruct predictions slice-by-slice

result = cell(1,nSlices);
for i = 1:nSlices
    mask = DataForStat.bwMaskLowFinal{i};
    img  = zeros(size(DataForStat.bwMaskLowFinal{i}));
    lbls = preds_cnn(sliceStarts(i):sliceEnds(i));
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
    colormap('jet'); colorbar; clim([0 1]); axis off;
    title(DataForStat.cases{i}, 'FontName','Arial');
end
sgtitle('Reconstructed - CNN Regression (BO)','FontName','Arial');
saveas(gcf, 'image/cnn regression/reconstruction_cnn_T1D_BO.png');

%% ===== Local functions =====

function layers = buildLayers(fcUnits, dropRate)
% CNN backbone: we keep conv blocks fixed, and tune FC units + dropout.
layers = [
    imageInputLayer([50 50 1],'Normalization','rescale-zero-one')

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer

    globalAveragePooling2dLayer
    fullyConnectedLayer(fcUnits)
    dropoutLayer(dropRate)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];
end

function obj = cnnRegObjFcn(T, XTr, YTr, XVal, YVal)
% Objective for BO: train on (XTr,YTr), return validation MSE on (XVal,YVal).
layers = buildLayers(T.fc, T.drop);
opts = trainingOptions(string(T.opt), ...
    'InitialLearnRate', T.initLR, ...
    'L2Regularization', T.l2, ...
    'MiniBatchSize', T.mb, ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5, ...
    'ExecutionEnvironment','auto', ...
    'Verbose', false);

try
    net = trainNetwork(XTr, YTr, layers, opts);
    YP = predict(net, XVal);
    YP = squeeze(YP);          % ensure column vector
    YVal = squeeze(YVal);      % ensure column vector
    obj = mean((YP - YVal).^2);
catch
    % Avoid using ME.message to prevent version-specific issues.
    warning('Trial failed. Assigning a large objective value.');
    obj = 1e6; % large penalty so BO avoids this region
end
end
