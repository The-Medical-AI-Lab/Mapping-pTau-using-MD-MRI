
clc; clear; close all;

%% Load dataset

dataFile = "dataset/classification_image_data_T1D_binary.mat";
S = load(dataFile);

X4 = S.X; % 50x50xCxN
y = double(S.yBinary(:)); % N×1, 0/1
thr = S.thresh; % threshold

fprintf("Loaded: X=%s, y=%s, thresh=%s\n", mat2str(size(X4)), mat2str(size(y)), num2str(thr));

sz = size(X4);
if numel(sz) ~= 4 || sz(1)~=50 || sz(2)~=50
    error("Expected X to be 50x50xCxN. Got %s", mat2str(sz));
end

C = sz(3);
N = sz(4);
if N ~= numel(y)
    error("Mismatch: size(X,4)=%d but length(y)=%d", N, numel(y));
end
fprintf("Interpreting X as 50x50xCxN with C=%d, N=%d\n", C, N);

idx0 = (y==0);
idx1 = (y==1);
if ~any(idx0) || ~any(idx1)
    error("One class is empty. class0=%d, class1=%d", sum(idx0), sum(idx1));
end

%% Prototype distributions

P0 = mean(X4(:,:,:,idx0), 4);
P1 = mean(X4(:,:,:,idx1), 4);

% Normalize each channel prototype to sum to 1
% for c = 1:C
%     P0c = double(P0(:,:,c));
%     P0(:,:,c) = P0c / sum(P0c(:));
%     P1c = double(P1(:,:,c));
%     P1(:,:,c) = P1c / sum(P1c(:));
% end

fprintf("\nPrototype normalization check (sum over 50x50):\n");
fprintf("P0: ");
fprintf("%.6f ", squeeze(sum(P0,[1 2])));
fprintf("\n");
fprintf("P1: ");
fprintf("%.6f ", squeeze(sum(P1,[1 2])));
fprintf("\n");

%% Optimal Transport settings

% eps: entropic regularization (bigger => smoother/faster, smaller => closer to true OT but slower)
ot.eps = 0.1;
ot.iters = 100;
ot.tol = 1e-6;  % early-stop based on change in u (set 0 to disable)
ot.useSingle = true;  % set true for speed/memory
ot.scaleAxes = true;  % scale x/y axes to comparable variance

% path
outDir = "image/optimal transport";

%% Build OT cost/kernel ONCE

fprintf("\n[OT] Building cost matrix/kernel\n");
[n1,n2] = deal(50,50);
[Cmat, Kker, Aker] = buildOTKernel(n1, n2, ot.eps, ot.scaleAxes, ot.useSingle);
% Aker = (K .* C) precomputed for cost evaluation without forming Gamma

fprintf("[OT] Ready. eps=%.4g, iters=%d, single=%d\n", ot.eps, ot.iters, ot.useSingle);

%% Compute OT distances to BOTH prototypes

% OTD(i,1) = OT(Pi, P0)
% OTD(i,2) = OT(Pi, P1)
OTD = zeros(N,2);
% disp(N);

for i = 1:N
    % disp(i)
    fprintf('[OT] %d / %d (%.1f%%)\n', i, N, 100*i/N);

    ot_to0 = zeros(C,1);
    ot_to1 = zeros(C,1);

    for c = 1:C
        Pi = double(X4(:,:,c,i));
        Q0 = double(P0(:,:,c));
        Q1 = double(P1(:,:,c));

        ot_to0(c) = sinkhornOT2D(Pi, Q0, Kker, Aker, ot.iters, ot.tol);
        ot_to1(c) = sinkhornOT2D(Pi, Q1, Kker, Aker, ot.iters, ot.tol);
    end

    OTD(i,1) = mean(ot_to0);
    OTD(i,2) = mean(ot_to1);

    if mod(i, max(1,round(N/10)))==0
        fprintf("[OT] %d/%d done\n", i, N);
    end
end

%% Margin: AD - CN (same sign convention as KL/JS margin)

OTD_margin = OTD(:,2) - OTD(:,1);

fprintf("\n[OT] Summary (to both prototypes)\n");

fprintf("Class0: OT->0 mean=%.6f (min=%.6f, max=%.6f), OT->1 mean=%.6f (min=%.6f, max=%.6f)\n", ...
    mean(OTD(idx0,1)), min(OTD(idx0,1)), max(OTD(idx0,1)), ...
    mean(OTD(idx0,2)), min(OTD(idx0,2)), max(OTD(idx0,2)));

fprintf("Class1: OT->0 mean=%.6f (min=%.6f, max=%.6f), OT->1 mean=%.6f (min=%.6f, max=%.6f)\n", ...
    mean(OTD(idx1,1)), min(OTD(idx1,1)), max(OTD(idx1,1)), ...
    mean(OTD(idx1,2)), min(OTD(idx1,2)), max(OTD(idx1,2)));

%% Visualization: OT margin

fig = figure('Units','inches','Position',[1 1 6 5],'Color','w');
boxplot(OTD_margin, y);
yline(0,'k--');
title("T1D Optimal Transport Margin (AD − CN)");
xlabel("Class (0=CN, 1=AD)");
ylabel("OT Margin");
set(fig,'PaperPositionMode','auto');
print(fig, fullfile(outDir,'Boxplot_OT_Margin_T1D.png'), '-dpng','-r600');
% close(fig);

%% Save results

save("dataset/classification_image_data_T1D_binary.mat", 'OTD', 'OTD_margin', '-append');
save("dataset/classification_data_T1D_binary.mat", 'OTD', 'OTD_margin', '-append');
fprintf("\nSaved OTD and OTD_margin to both binary .mat files.\n");

%% Functions

function [Cmat, K, A] = buildOTKernel(n1, n2, eps, scaleAxes, useSingle)
% buildOTKernel:
%   - Create grid coordinates for an n1-by-n2 image
%   - Cmat: squared Euclidean cost between bins
%   - K: Sinkhorn kernel exp(-C/eps)
%   - A: elementwise product (K .* C) used for fast cost computation:
%   cost = u' * ( (K.*C) * v ) = u' * (A * v)

    [Xg,Yg] = meshgrid(1:n2, 1:n1);  % X: col, Y: row
    coords = [Xg(:), Yg(:)];  % n x 2

    if scaleAxes
        % Scale axes to comparable variance so x/y contribute similarly
        sx = std(coords(:,1)); sy = std(coords(:,2));
        coords(:,1) = coords(:,1) / max(sx, eps);
        coords(:,2) = coords(:,2) / max(sy, eps);
    end

    Cmat = pdist2(coords, coords, 'euclidean').^2;  % n x n
    K = exp(-Cmat / eps);

    % numerical floor
    K = max(K, realmin);

    % Precompute A = K .* C for fast OT cost evaluation
    A = K .* Cmat;

    if useSingle
        Cmat = single(Cmat);
        K = single(K);
        A = single(A);
    end
end


function cost = sinkhornOT2D(P2, Q2, K, A, nIter, tol)
% sinkhornOT2D:
%   Compute entropic-regularized OT cost between two 2D distributions P2 and Q2.
%   Input: P2, Q2 are n1-by-n2 matrices (here 50x50), nonnegative, ideally sum to 1.
%   Output: OT cost (approx Wasserstein-2 with entropic regularization)
%
% Key trick:
%   We do NOT build Gamma. Cost = sum_ij (u_i K_ij v_j) C_ij = u' * ((K.*C) * v) = u' * (A * v).

    a = double(P2(:));
    b = double(Q2(:));

    % Ensure proper probability vectors
    a = max(a, realmin);
    a = a / sum(a);
    b = max(b, realmin);
    b = b / sum(b);

    if ~isa(K, 'double')
        % match precision with kernel for speed
        a = cast(a, 'like', K);
        b = cast(b, 'like', K);
    end

    u = ones(size(a), 'like', K);
    v = ones(size(b), 'like', K);
    
    KT = K';

    for it = 1:nIter
        % disp(it);

        u_prev = u;

        Kv = K * v;
        u = a ./ max(Kv, realmin('like', K));

        KTu = KT * u;
        v = b ./ max(KTu, realmin('like', K));

        if tol > 0
            rel = norm(double(u - u_prev), 1) / max(norm(double(u_prev),1), realmin);
            if rel < tol
                break;
            end
        end
    end

    % OT cost using precomputed A = K.*C
    cost = double(u' * (A * v));
end
