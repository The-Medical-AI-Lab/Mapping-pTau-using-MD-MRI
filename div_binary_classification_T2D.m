
clc; clear; close all;

%% Load the dataset

dataFile = "dataset/classification_image_data_T2D_binary.mat";
S = load(dataFile);

X4 = S.X; % 4-D single
y  = double(S.yBinary(:)); % N×1, 0/1
thr = S.thresh; % threshold
fprintf("Loaded: X=%s, y=%s, thresh=%.4f\n", mat2str(size(X4)), mat2str(size(y)), thr);

%% Check X has the right size

sz = size(X4);
if numel(sz) ~= 4
    error("S.X is not 4-D. Got size=%s", mat2str(sz));
end

if sz(1)~=50 || sz(2)~=50
    error("Expected X to start with 50x50. Got size=%s", mat2str(sz));
end

C = sz(3);
N = sz(4);

if N ~= numel(y)
    error("Mismatch: size(X,4)=%d but length(yBinary)=%d", N, numel(y));
end

fprintf("Interpreting X as 50x50xCxN with C=%d, N=%d\n", C, N);

%% Check whether original distributions sum to 1

Xraw = double(X4);
epsTol = 1e-6;

sz = size(Xraw);
C = sz(3);
N = sz(4);

sumErr = zeros(N, C);

% disp(size(Xraw));
% disp(size(sumErr));

for c = 1:C
    for i = 1:N
        s = sum(Xraw(:,:,c,i), 'all');
        sumErr(i,c) = abs(s - 1);
    end
end

maxErr = max(sumErr(:));
meanErr = mean(sumErr(:));
pctBad = mean(sumErr(:) > epsTol) * 100;

fprintf('\nProbability normalization check\n');
fprintf('Max |sum-1|: %.3e\n', maxErr);
fprintf('Mean |sum-1|: %.3e\n', meanErr);

%% Calculate the prototype (average distribution) of each class

idx0 = (y==0);
idx1 = (y==1);

if ~any(idx0) || ~any(idx1)
    error("One class is empty. class0=%d, class1=%d", sum(idx0), sum(idx1));
end

P0 = mean(X4(:,:,:,idx0), 4);
P1 = mean(X4(:,:,:,idx1), 4);

for c = 1:C
    P0c = P0(:,:,c);
    P0(:,:,c) = P0c / sum(P0c(:));
    P1c = P1(:,:,c);
    P1(:,:,c) = P1c / sum(P1c(:));
end

%% Check if Prototypes sum to 1

sum_P0 = squeeze(sum(P0, [1 2])); 
sum_P1 = squeeze(sum(P1, [1 2]));

fprintf('Sums for P0 (Class 0):\n');
fprintf('%.15f ', sum_P0); 
fprintf('\n');

fprintf('Sums for P1 (Class 1):\n');
fprintf('%.15f ', sum_P1); 
fprintf('\n');

%% Plot Prototypes

% Path
saveDir = "image/divergence";

globalMax = max([P0(:); P1(:)]);
globalMin = min([P0(:); P1(:)]);

for c = 1:C
    Proto_CN = P0(:,:,c);
    Proto_AD = P1(:,:,c);
    Proto_Diff = Proto_AD - Proto_CN;

    disp(sum(Proto_CN(:)))
    disp(sum(Proto_AD(:)))
    disp(sum(Proto_Diff(:)))
    
    figure('Units', 'inches', 'Position', [1, 1, 15, 5], 'Color', 'w'); 
    
    % Subplot 1: pTau=0 (Class 0)
    subplot(1, 3, 1);
    contourf(Proto_CN);
    title('pTau=0 (Average)'); 
    ylabel('T_2 [ms]'); 
    xlabel('MD [\mum^2/ms]'); 
    grid on;
    axis equal;
    clim([globalMin, globalMax]);
    colorbar;
    
    % Subplot 2: pTau=1 (Class 1)
    subplot(1, 3, 2);
    contourf(Proto_AD); 
    title('pTau=1 (Average)'); 
    ylabel('T_2 [ms]'); 
    xlabel('MD [\mum^2/ms]'); 
    grid on;
    axis equal;
    clim([globalMin, globalMax]);
    colorbar;

    % Difference: pTau=1 - pTau=0
    subplot(1, 3, 3);
    contourf(Proto_Diff, 'LineColor', 'none');
    title('pTau=1 (Ave) − pTau=1 (Ave)');
    xlabel('MD [\mum^2/ms]');
    ylabel('T_2 [ms]');
    axis equal;
    grid on;
    clim([globalMin, globalMax]);
    colorbar;
    
    sgtitle('Average T2-MD Joint Distributions and Difference', 'FontName', 'Arial');
    
    set(gcf, 'PaperPositionMode', 'auto');
    outFile = fullfile(saveDir, 'PsiT2D_Average_binary.png');
    print(gcf, outFile, '-dpng', '-r600');
    fprintf('Saved: %s\n', outFile);
    
    close(gcf);
end

%% Divergence: KL Divergence and JS Divergence

KLD = zeros(N,2);   % col1: to P0, col2: to P1
JSD = zeros(N,2);

for i = 1:N
    kl_to0  = zeros(C,1);
    kl_to1  = zeros(C,1);
    js_to0  = zeros(C,1);
    js_to1  = zeros(C,1);

    for c = 1:C
        Pi = double(X4(:,:,c,i));

        Q0 = double(P0(:,:,c));
        Q1 = double(P1(:,:,c));

        kl_to0(c) = KLdiv(Pi, Q0);
        kl_to1(c) = KLdiv(Pi, Q1);

        js_to0(c) = JSdiv(Pi, Q0);
        js_to1(c) = JSdiv(Pi, Q1);
    end

    KLD(i,1) = mean(kl_to0);
    KLD(i,2) = mean(kl_to1);

    JSD(i,1) = mean(js_to0);
    JSD(i,2) = mean(js_to1);
end

fprintf("\nDivergence summary (to both prototypes)\n");
fprintf("Class0: mean(KL to0)=%.6f, mean(KL to1)=%.6f\n", mean(KLD(idx0,1)), mean(KLD(idx0,2)));
fprintf("Class1: mean(KL to0)=%.6f, mean(KL to1)=%.6f\n", mean(KLD(idx1,1)), mean(KLD(idx1,2)));
fprintf("Class0: mean(JS to0)=%.6f, mean(JS to1)=%.6f\n", mean(JSD(idx0,1)), mean(JSD(idx0,2)));
fprintf("Class1: mean(JS to0)=%.6f, mean(JS to1)=%.6f\n", mean(JSD(idx1,1)), mean(JSD(idx1,2)));

% Save
targetFile = "dataset/classification_image_data_T2D_binary.mat";
save(targetFile, 'KLD', 'JSD', '-append');
fprintf('Variables KLD and JSD have been appended to:\n%s\n', targetFile);
info = whos('-file', targetFile);
disp({info.name}');

targetFile = "dataset/classification_data_T2D_binary.mat";
save(targetFile, 'KLD', 'JSD', '-append');
fprintf('Variables KLD and JSD have been appended to:\n%s\n', targetFile);
info = whos('-file', targetFile);
disp({info.name}');

%% Visaulization

% Margin definition:
% positive -> closer to CN prototype
% negative -> closer to AD prototype
% zero -> equidistant to both
% AD - CN
KLD_margin = KLD(:,2) - KLD(:,1);
JSD_margin = JSD(:,2) - JSD(:,1);

outDir = "image/divergence";

% JSD margin
fig1 = figure('Units','inches','Position',[1 1 6 5],'Color','w');
boxplot(JSD_margin, y);
yline(0,'k--');
title("T2D Jensen–Shannon Divergence Margin (pTau=1 − pTau=0)");
xlabel("Class (pTau=0, pTau=1)");
ylabel("JSD Margin");
set(fig1,'PaperPositionMode','auto');
print(fig1, fullfile(outDir,'Boxplot_JSD_Margin_T2D.png'), '-dpng','-r600');

% KLD margin
fig2 = figure('Units','inches','Position',[1 1 6 5],'Color','w');
boxplot(KLD_margin, y);
yline(0,'k--');
title("T2D Kullback–Leibler Divergence Margin (pTau=1 − pTau=0)");
xlabel("Class (pTau=0, pTau=1)");
ylabel("KLD Margin");
set(fig2,'PaperPositionMode','auto');
print(fig2, fullfile(outDir,'Boxplot_KLD_Margin_T2D.png'), '-dpng','-r600');

%% functions

function kl = KLdiv(P,Q)
    eps0 = 1e-12;
    P = P + eps0; Q = Q + eps0;
    P = P / sum(P(:)); Q = Q / sum(Q(:));
    kl = sum(P(:).*log(P(:)./Q(:)));
end

function js = JSdiv(P,Q)
    eps0 = 1e-12;
    P = P + eps0; Q = Q + eps0;
    P = P / sum(P(:)); Q = Q / sum(Q(:));
    M = 0.5*(P+Q);
    js = 0.5*sum(P(:).*log(P(:)./M(:))) + 0.5*sum(Q(:).*log(Q(:)./M(:)));
end
