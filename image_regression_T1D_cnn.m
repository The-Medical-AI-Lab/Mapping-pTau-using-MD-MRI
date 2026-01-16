
clc; clear; close all;

%% 1. Load dataset and struct

load("dataset/regression_image_data_T1D.mat", 'X', 'y');  % X: 50x50x1xN, y: N×1
load('dataset/DataForStat_101124.mat');  % For voxel reconstruction
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

%% 4. 5-Fold Cross-Validation with CNN Regression

N = size(X,4);
k = 5;
cv = cvpartition(N, 'KFold', k);
mse_cnn = zeros(k,1);
r2_cnn  = zeros(k,1);
preds_cnn = zeros(N,1);

for fold = 1:k
    fprintf('\n=== Fold %d/%d ===\n', fold, k);

    tr = training(cv, fold);
    te = test(cv, fold);

    XTrain = X(:,:,:,tr);
    YTrain = y(tr);
    XTest  = X(:,:,:,te);
    YTest  = y(te);

    layers = [
        imageInputLayer([50 50 1],'Normalization','rescale-zero-one')
        convolution2dLayer(3,64,'Padding','same');
        batchNormalizationLayer;
        reluLayer;
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,128,'Padding','same');
        batchNormalizationLayer;
        reluLayer;
        maxPooling2dLayer(2,'Stride',2)
        convolution2dLayer(3,256,'Padding','same');
        batchNormalizationLayer;
        reluLayer;
        globalAveragePooling2dLayer
        fullyConnectedLayer(64);
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer
        ];

    options = trainingOptions('adam', 'MaxEpochs', 20, 'InitialLearnRate', 1e-3, 'MiniBatchSize', 128, 'Shuffle', 'every-epoch', 'Verbose', true);

    net = trainNetwork(XTrain, YTrain, layers, options);

    YPred = predict(net, XTest);
    preds_cnn(te) = YPred;

    mse_cnn(fold) = mean((YPred - YTest).^2);
    SSres = sum((YTest - YPred).^2);
    SStot = sum((YTest - mean(YTest)).^2);
    r2_cnn(fold) = 1 - SSres/SStot;

    fprintf('Fold %d - MSE: %.4f, R^2: %.4f\n', fold, mse_cnn(fold), r2_cnn(fold));
end

%% 5. Display average metrics

fprintf('\n===== CNN Regression (5-Fold CV) Performance =====\n');
fprintf('MSE: %.4f ± %.4f\n', mean(mse_cnn), std(mse_cnn));
fprintf('R^2: %.4f ± %.4f\n', mean(r2_cnn), std(r2_cnn));

%% 6. Plot metrics with error bars

figure;
bar([mean(mse_cnn), mean(r2_cnn)]); hold on;
errorbar(1:2, [mean(mse_cnn), mean(r2_cnn)], [std(mse_cnn), std(r2_cnn)], '.k', 'LineWidth', 1.5);
set(gca,'XTickLabel',{'MSE','R^2'});
title('CNN Regression - 5 Fold CV Metrics');
grid on;
saveas(gcf, 'image/cnn regression/performance_crossvalidation_T1D.png');

figure;

% MSE
subplot(1,2,1);
bar(mean(mse_cnn), 'FaceColor', [0.2 0.6 0.5]); hold on;
errorbar(1, mean(mse_cnn), std(mse_cnn), '.k', 'LineWidth', 1.5);
set(gca, 'XTick', 1, 'XTickLabel', {'MSE'});
ylabel('Score');
title('CNN Regression - MSE');
grid on;

% R-square
subplot(1,2,2);
bar(mean(r2_cnn), 'FaceColor', [0.4 0.4 0.8]); hold on;
errorbar(1, mean(r2_cnn), std(r2_cnn), '.k', 'LineWidth', 1.5);
set(gca, 'XTick', 1, 'XTickLabel', {'R^2'});
ylabel('Score');
title('CNN Regression - R^2');
grid on;

% Save the plot
saveas(gcf, 'image/cnn regression/performance_crossvalidation_T1D.png');

%% 7. Reconstruct predictions slice-by-slice

result = cell(1,nSlices);
for i = 1:nSlices
    mask = DataForStat.bwMaskLowFinal{i};
    img  = zeros(size(mask));
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
    colormap('jet');
    colorbar;
    clim([0 1]);
    axis off;
    title(DataForStat.cases{i}, 'FontName','Arial');
end
sgtitle('Reconstructed - CNN Regression','FontName','Arial');
saveas(gcf, 'image/cnn regression/reconstruction_cnn_T1D.png');
