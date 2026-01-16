
clc; clear; close all;

%% 1. Load dataset and struct

load("dataset/classification_image_data_T2D_binary.mat");  % X: 50x50x1xN, yBinary: N×1
y = yBinary;
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

%% 4. 5-Fold Cross-Validation with CNN

N = size(X,4);
k = 5;
cv = cvpartition(y, 'KFold', k);
acc_cnn = zeros(k,1);
prec_cnn = zeros(k,1);
rec_cnn = zeros(k,1);
kappa_cnn = zeros(k,1);
preds_cnn = zeros(N,1);  % full test predictions

for fold = 1:k
    tr = training(cv, fold);
    te = test(cv, fold);

    % Train/Test sets
    XTrain = X(:,:,:,tr);
    YTrain = categorical(y(tr), [0 1], {'neg','pos'});
    XTest  = X(:,:,:,te);
    YTest  = categorical(y(te), [0 1], {'neg','pos'});

    % Define CNN architecture
    layers = [
        imageInputLayer([50 50 1],'Normalization','rescale-zero-one','Name','input')
        convolution2dLayer(3,64,'Padding','same','Name','conv1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        maxPooling2dLayer(2,'Stride',2,'Name','pool1')

        convolution2dLayer(3,128,'Padding','same','Name','conv2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        maxPooling2dLayer(2,'Stride',2,'Name','pool2')

        convolution2dLayer(3,256,'Padding','same','Name','conv3')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','relu3')
        globalAveragePooling2dLayer('Name','gap')

        fullyConnectedLayer(2,'Name','fc')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')
    ];

    options = trainingOptions('adam', 'MaxEpochs', 20, 'InitialLearnRate', 1e-3, 'MiniBatchSize', 128, 'Shuffle', 'every-epoch', 'Verbose', false);

    net = trainNetwork(XTrain, YTrain, layers, options);

    % Predict & compute metrics
    YPred = classify(net, XTest);
    preds_cnn(te) = double(YPred=='pos');

    [acc_cnn(fold), prec_cnn(fold), rec_cnn(fold), kappa_cnn(fold)] = computeMetrics(double(YTest=='pos'), double(YPred=='pos'));
end

%% 5. Display average metrics

fprintf('\n===== CNN (5-Fold CV) Performance =====\n');
fprintf('Acc:   %.3f ± %.3f\n', mean(acc_cnn), std(acc_cnn));
fprintf('Prec:  %.3f ± %.3f\n', mean(prec_cnn), std(prec_cnn));
fprintf('Rec:   %.3f ± %.3f\n', mean(rec_cnn), std(rec_cnn));
fprintf('Kappa: %.3f ± %.3f\n', mean(kappa_cnn), std(kappa_cnn));

%% 6. Plot metrics with error bars

metrics_mean = [mean(acc_cnn), mean(prec_cnn), mean(rec_cnn), mean(kappa_cnn)];
metrics_std  = [std(acc_cnn), std(prec_cnn), std(rec_cnn), std(kappa_cnn)];
metric_names = {'Accuracy','Precision','Recall','Kappa'};

figure;
bar(metrics_mean); hold on;
errorbar(1:4, metrics_mean, metrics_std, '.k', 'LineWidth', 1.5);
set(gca,'XTickLabel',metric_names);
ylim([0 1]); grid on;
ylabel('Performance');
title('CNN Classification - 5 Fold CV Metrics');
saveas(gcf, 'image/cnn classification/performance_cv_cnn_binary_T2D.png');

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
sgtitle('Reconstructed - CNN Classification','FontName','Arial');
saveas(gcf, 'image/cnn classification/reconstruction_cnn_binary_T2D.png');

%% Helper Function: computeMetrics

function [ac, pr, re, ka] = computeMetrics(yT, yP)
    C = confusionmat(yT, yP);
    TN = C(1,1); FP = C(1,2); FN = C(2,1); TP = C(2,2);
    ac = (TP+TN)/sum(C(:)); pr = TP/(TP+FP); re = TP/(TP+FN);
    Po = (TP+TN)/sum(C(:)); Pe = ((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(sum(C(:))^2);
    ka = (Po-Pe)/(1-Pe);
end
