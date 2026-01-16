
clc; clear; close all;

%% Load data

% Load the binary classification dataset of T2D
load("dataset/classification_data_T2D_binary.mat");
y = yBinary;

% Print the size of X and y
fprintf('Size of X: %d × %d\n', size(X,1), size(X,2));
fprintf('Size of y: %d × %d\n\n', size(y,1), size(y,2));

%% PCA + Loading Maps (retain >=95% variance)

% 1) PCA
[coeff, ~, ~, ~, explained] = pca(X, 'Centered', true);
cumExpl = cumsum(explained);
nComp   = find(cumExpl >= 95, 1, 'first');   % First PC index reaching >=95% variance
K = max(1, nComp);                           % Ensure at least 1 component

% 2) Reshape size for each loading map (modify according to your feature layout)
imgH = 50; 
imgW = 50;

% 3) Set a symmetric color scale across all selected PCs
coef_sel = coeff(:,1:K);
v = max(abs(coef_sel), [], 'all');
cax = [-v, v];

% 4) Layout (rows × cols)
% rows = ceil(sqrt(K));
% cols = ceil(K / rows);
rows = 3;
cols = 7;

% 5) Plot settings (width & height in inches; final resolution = DPI × inches)
W_in = 18; 
H_in = 6;
fig = figure('Units','inches','Position',[1 1 W_in H_in]);
tiledlayout(rows, cols, 'Padding','compact', 'TileSpacing','compact');
colormap('jet')

for k = 1:K
    nexttile;
    Lk = reshape(coef_sel(:,k), imgH, imgW); % Reshape vector to image
    imagesc(Lk);
    axis image off;
    clim([-0.5 0.5]);
    cb = colorbar; 
    cb.Location = 'eastoutside';
    title(sprintf('PC %d Loading Map\n(%.2f%% variance)', k, explained(k)));
end

sgtitle(sprintf('PCA Component Loading Maps (T2D)', K));

% 6) Save figure (pixels = resolution × inches)
saveas(fig, 'image/pca/pca_loading_maps_T2D.png');

%% Variance Explained Plot for ALL samples

figVarAll = figure('Name','All – PCA variance','Position',[100 100 800 500]);

yyaxis left
bar(explained(1:20)); % Plot first 20 PCs
xlabel('Principal component index');
ylabel('Individual variance (%)');
title('All Samples – Variance Explained (T2D)');

yyaxis right
plot(cumExpl(1:20), '-o', 'LineWidth',1.2);
ylabel('Cumulative variance (%)');

hold on
xline(nComp, '--r', sprintf('PC = %d', nComp), 'LineWidth',1.2);
yline(95, '--', '95 %', 'LineWidth',1);
grid on

% Save figure
saveas(figVarAll, fullfile('image/pca/', 'PCA_variance_all_T2D_classification.png'));

%% Split samples by label

idxPos = (y == 1); % Positive class
idxNeg = (y == 0); % Negative class

X_pos = X(idxPos, :);
X_neg = X(idxNeg, :);

fprintf('Positive samples: %d\n', nnz(idxPos));
fprintf('Negative samples: %d\n\n', nnz(idxNeg));

%% Run PCA and plot results for each group

[coeffPos, scorePos, nPos] = runPCAandPlot(X_pos, 'Positive');
[coeffNeg, scoreNeg, nNeg] = runPCAandPlot(X_neg, 'Negative');

%% Function Definition

function [coeff_trunc, score_trunc, nComp] = runPCAandPlot(Xgroup, tag)

% runPCAandPlot  Perform PCA on Xgroup, retain ≥95 % variance, produce variance-explained plots and loading maps.
%
% Inputs
%   Xgroup : [N × D] matrix of samples in one class
%   tag    : 'Positive' or 'Negative' (used in figure titles / filenames)
%
% Outputs
%   coeff_trunc : [D × nComp] loadings for retained components
%   score_trunc : [N × nComp] PCA scores for retained components
%   nComp       : number of components explaining ≥95 % variance

    % PCA (principal component analysis)
    [coeff, score, ~, ~, explained, mu] = pca(Xgroup, 'Centered', true);
    cumExpl = cumsum(explained);
    threshold = 95;
    nComp = find(cumExpl >= threshold, 1, 'first');

    fprintf('%s: %d PCs explain %.2f %% variance\n', tag, nComp, cumExpl(nComp));

    % Variance-explained figure
    figVar = figure('Name',[tag ' – PCA variance'],'Position',[100 100 800 500]);

    yyaxis left
    bar(explained(1:20)); % First 20 components only
    xlabel('Principal component index');
    ylabel('Individual variance (%)');
    title([tag ' – Variance Explained (T2D)']);

    yyaxis right
    plot(cumExpl(1:20), '-o', 'LineWidth',1.2);
    ylabel('Cumulative variance (%)');

    hold on
    xline(nComp, '--r', sprintf('PC = %d', nComp), 'LineWidth',1.2);
    yline(95, '--', '95 %', 'LineWidth',1);
    grid on

    % Save figure
    saveas(figVar, fullfile('image/pca/', sprintf('PCA_variance_%s_T2D_classification.png', lower(tag))));

    % Loading-map figure
    imgH = 50; 
    imgW = 50;
    rows = ceil(sqrt(nComp));
    cols = ceil(nComp/rows);

    figLoad = figure('Name',[tag ' – PCA loadings'], 'Position',[100 100 300*cols 260*rows]);
    colormap jet

    for k = 1:nComp
        subplot(rows, cols, k);
        loadingMap = reshape(coeff(:,k), imgH, imgW);
        imagesc(loadingMap);
        colorbar;
        clim([-0.5 0.5]);
        axis image off
        title(sprintf('PC %d\n(%.1f%%)', k, explained(k)));
    end
    sgtitle([tag ' – Loading Maps (T2D)']);

    % Save figure
    saveas(figLoad, fullfile('image/pca/', sprintf('PCA_loadings_%s_T2D_classification.png', lower(tag))));

    % Output truncated coefficients and scores (optional downstream use)
    coeff_trunc = coeff(:,1:nComp);
    score_trunc = score(:,1:nComp);

end
