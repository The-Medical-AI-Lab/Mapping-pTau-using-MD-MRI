
clc; clear; close all;

%% The relevant struct fields are

% pTau_DAB_Low: co-registered pTau %-area map (eight slices total).
% Abeta_DAB_Low: co-registered Abeta %-area map (eight slices total).
% bwMaskLowFinal: binary mask for each slice, used to define regions of interest (ROI).
% PsiT1D: same for D-T1.
% PsiT2D: voxelwise 50 x 50 D-T2 density functions vectorized to save space (i.e., eight slices of 50 x 50 x [number of non-zero voxels in the slice]).
% row_pixCurFinal, col_pixCurFinal: row and column coordinates of valid voxel in bwMaskLowFinal.
% Should be using Abeta_DAB_Low for quantitative analyses. Abeta_DAB_Ref is just for visualization.
% All above are 1x8 cells.

%% Load the data

load("dataset/DataForStat_101124.mat");

%% View the size of dataset

% disp("Dataset Size:");
% size(DataForStat)
% 
% disp("Size of pTau_DAB_Low (Co-registered pTau %-area map):");
% size(DataForStat.pTau_DAB_Low)
% 
% disp("Size of Abeta_DAB_Low (Co-registered Abeta %-area map):");
% size(DataForStat.Abeta_DAB_Low)
% 
% disp("Size of PsiT1D (Voxel-wise 50x50 D-T1 density functions):");
% size(DataForStat.PsiT1D)
% 
% disp("Size of PsiT2D (Voxel-wise 50x50 D-T2 density functions):");
% size(DataForStat.PsiT2D)
% 
% disp("Size of bwMaskLowFinal (Binary mask for each slice):");
% size(DataForStat.bwMaskLowFinal)
% 
% disp("Size of row_pixCurFinal (Row coordinates of bwMaskLowFinal):");
% size(DataForStat.row_pixCurFinal)
% 
% disp("Size of row_pixCurFinal (Column coordinates of bwMaskLowFinal):");
% size(DataForStat.col_pixCurFinal)

%% To reconstruct the ith Psi(D,T2) as an image

i = 4; % range from 1 to 8 (8 slices), choose the 4 as an example

% Voxel means Volume Pixel (valid voxel and invalid or background voxel)
% Initialize the output array with the correct dimensions
% Im is initialized as an all zero matrix, whose size is based on the size of PsiT2D data and bwMaskLowFinal
% Im is 4 dimensional, 50x50x125x89
% Dimension 1 and 2: Same as the first two dimensions of PsiT2D{i} (50×50), representing the spatial dimension of the localized image.
% Dimensions 3 and 4: Same size as bwMaskLowFinal{i} (125×89), representing the spatial coordinates of the original MRI image.

Im = zeros([size(DataForStat.PsiT2D{i}, 1), size(DataForStat.PsiT2D{i}, 2), size(DataForStat.bwMaskLowFinal{i})]);

disp(size(Im(:,:,:,:))) % 50 50 125 89
disp(size(DataForStat.PsiT1D{i})) % 50 50 7452
disp(size(DataForStat.PsiT2D{i})) % 50 50 7452
disp(size(DataForStat.pTau_DAB_Low{i})) % 125 89
disp(size(DataForStat.Abeta_DAB_Low{i})) % 125 89
disp(size(DataForStat.row_pixCurFinal{i})) % 7452 1
disp(size(DataForStat.col_pixCurFinal{i})) % 7452 1
disp(size(DataForStat.bwMaskLowFinal{i})) % 125 89
disp(sum(DataForStat.bwMaskLowFinal{i}(:))); % 7452 ROI

% So the sum of all 1 in bwMaskLowFinal is valid voxel
% White (value of 1) represent the region of interest (ROI)
% Black (value of 0) represent a background or invalid area

% Loop through each element in the row_pixCurFinal and col_pixCurFinal arrays
% According to the coordinates given by row_pixCurFinal {i} and col_pixCurFinal {i}
% map the 50 × 50 dimensional data corresponding to PsiT2D to the (row, col) position of the MRI image
for j = 1:length(DataForStat.row_pixCurFinal{i})
    % Traverse the positions of all valid voxels in the ROI
    row = DataForStat.row_pixCurFinal{i}(j);
    col = DataForStat.col_pixCurFinal{i}(j);
    % Extract the jth 50x50 segment of the PsiT2D{i}
    Im(:,:,row, col) = squeeze(DataForStat.PsiT2D{i}(:,:,j));
end

%% Visaulize ROI (region of interest) for each slice, and we only use vaild voxels to construct new dataset

figure('Name','All ROI Masks','NumberTitle','off','Units', 'inches', 'Position', [1, 1, 24, 5]);
valid_amount = 0;
valid_num = [];

for i = 1:8
    % bwMaskLow: initial binary mask from low‐threshold segmentation
    % bwMaskLowFinal: final ROI mask after morphological cleaning and connected‐component filtering
    mask = DataForStat.bwMaskLow{i};
    [H, W] = size(mask);
    validPixelCount = sum(mask(:));
    valid_amount = valid_amount + validPixelCount;
    % disp(validPixelCount);
    valid_num = [valid_num, validPixelCount];

    subplot(1,8,i);
    imagesc(mask);
    colormap(gray);
    % axis image off; % image axis
    title(sprintf('%d x %d \n %d valid voxels', H, W, validPixelCount), 'FontSize', 12);
end

sgtitle('Initial ROI Mask', 'FontName', 'Arial'); % Add title

% disp(valid_amount)

% Display the array of valid voxel counts
% disp('Valid voxel counts per slice:');
% disp(valid_num);
% disp(sum(valid_num));

% Plot the Binary Mask of each slice
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'ROI_Masks_bwMaskLow.png'), '-dpng', '-r600');

%% Visaulize ROI (region of interest) for each slice, and we only use vaild voxels to construct new dataset

figure('Name', 'ROI Masks', 'NumberTitle', 'off', 'Units', 'inches', 'Position', [1, 1, 24, 5]);
valid_amount = 0;
valid_num = [];

for i = 1:8
    % bwMaskLow: initial binary mask from low‐threshold segmentation
    % bwMaskLowFinal: final ROI mask after morphological cleaning and connected‐component filtering
    mask = DataForStat.bwMaskLowFinal{i};
    [H, W] = size(mask);
    validPixelCount = sum(mask(:));
    valid_amount = valid_amount + validPixelCount;
    % disp(validPixelCount);
    valid_num = [valid_num, validPixelCount];

    subplot(1,8,i);
    imagesc(mask);
    colormap(gray);
    % axis image off;
    title(sprintf('%d x %d \n %d valid voxels', H, W, validPixelCount), 'FontSize', 12);
end

sgtitle('Final ROI Mask', 'FontName', 'Arial'); % Add title

% disp(valid_amount)

% Display the array of valid voxel counts
% disp('Valid voxel counts per slice:');
% disp(valid_num);
% disp(sum(valid_num));

% Plot the Binary Mask of each slice
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'ROI_Masks_bwMaskLowFinal.png'), '-dpng', '-r600');

%% Plot contours of joint distribution of some sample valid voxels

% figure('Units','inches','Position',[1 1 12 12]);  % [x, y, width, height]
% plotCount = 1;
% for row = 1:size(Im, 3)
%     for col = 1:size(Im, 4)
%         if nnz(Im(:, :, row, col)) > 0
%             if plotCount > 25
%                 break;
%             end
%             subplot(5, 5, plotCount);
%             contourf(Im(:, :, row, col)); % Draw contour lines
%             title(['Row: ', num2str(row), ', Col: ', num2str(col)]);
%             plotCount = plotCount + 1;
%         end
%     end
% end

%% MRI Ref

% This is to create a figure panel using the histology stained images

% MATLAB script to plot pTau_DAB_ref for i side by side
% Indices to plot
indices = [1, 2, 3, 4, 5, 6, 7, 8];

% Create a new figure
% Width: 24 inches, Height: 5 inches
figure('Units', 'inches', 'Position', [1, 1, 24, 5]);

% Loop through the specified indices
% MRI_Ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    % imagesc(DataForStat.pTau_DAB_Low{i}); colormap('jet'); % Plot the data
    % imagesc(DataForStat.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat.MRI_Ref{i}, []); % Plot the data, display the i-th MRI image, this is imshow not imagesc
    imagesc(DataForStat.MRI_Ref{i}); colormap('gray'); % This is imagesc not imshow

    % Add colorbar to show value scale
    colorbar;
    % clim([0 1]);

    axis off; % Remove axes
    label = DataForStat.cases{i}; 
    if label =='CN'
        title('CN', 'FontName', 'Arial'); % Add a title with Arial font
    elseif label=='AD'
        title('AD', 'FontName', 'Arial'); % Add a title with Arial font
    end
end

% Adjust layout for better visualization
% sgtitle('Hyperphosphorylated Tau', 'FontName', 'Arial'); % Add title
% sgtitle('Amyloid Beta', 'FontName', 'Arial'); % Add title
sgtitle('Reference MRI', 'FontName', 'Arial'); % Add title

% Save the plot
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'MRI_Ref.png'), '-dpng', '-r600');

%% Hyperphosphorylated Tau Low

% This is to create a figure panel using the histology stained images

% MATLAB script to plot pTau_DAB_ref for i side by side
% Indices to plot
indices = [1, 2, 3, 4, 5, 6, 7, 8];

% Create a new figure
% Width: 24 inches, Height: 5 inches
figure('Units', 'inches', 'Position', [1, 1, 24, 5]);

% Loop through the specified indices
% pTau_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    imagesc(DataForStat.pTau_DAB_Low{i}); colormap('jet'); % Plot the data
    % imagesc(DataForStat.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat.MRI_Ref{i},[]); % Plot the data, display the i-th MRI image

    % Add colorbar to show value scale
    colorbar;
    clim([0 1]);

    axis off; % Remove axes
    label = DataForStat.cases{i}; 
    if label =='CN'
        title('CN', 'FontName', 'Arial'); % Add a title with Arial font
    elseif label=='AD'
        title('AD', 'FontName', 'Arial'); % Add a title with Arial font
    end
end

% Adjust layout for better visualization
sgtitle('Hyperphosphorylated Tau Low', 'FontName', 'Arial'); % Add title
% sgtitle('Amyloid Beta', 'FontName', 'Arial'); % Add title
% sgtitle('Reference MRI', 'FontName', 'Arial'); % Add title

% Save the plot
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'Hyperphosphorylated_Tau_Low.png'), '-dpng', '-r600');

%% Hyperphosphorylated Tau ref

% This is to create a figure panel using the histology stained images

% MATLAB script to plot pTau_DAB_ref for i side by side
% Indices to plot
indices = [1, 2, 3, 4, 5, 6, 7, 8];

% Create a new figure
% Width: 24 inches, Height: 5 inches
figure('Units', 'inches', 'Position', [1, 1, 24, 5]);

% Loop through the specified indices
% pTau_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    imagesc(DataForStat.pTau_DAB_ref{i}); colormap('jet'); % Plot the data
    % imagesc(DataForStat.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat.MRI_Ref{i},[]); % Plot the data, display the i-th MRI image

    % Add colorbar to show value scale
    colorbar;
    clim([0 1]);

    axis off; % Remove axes
    label = DataForStat.cases{i}; 
    if label =='CN'
        title('CN', 'FontName', 'Arial'); % Add a title with Arial font
    elseif label=='AD'
        title('AD', 'FontName', 'Arial'); % Add a title with Arial font
    end
end

% Adjust layout for better visualization
sgtitle('Hyperphosphorylated Tau ref', 'FontName', 'Arial'); % Add title
% sgtitle('Amyloid Beta', 'FontName', 'Arial'); % Add title
% sgtitle('Reference MRI', 'FontName', 'Arial'); % Add title

% Save the plot
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'Hyperphosphorylated_Tau_ref.png'), '-dpng', '-r600')

%% Amyloid Beta Low

% Width: 12 inches, Height: 3 inches
figure('Units', 'inches', 'Position', [1, 1, 24, 5]);

% Loop through the specified indices
% Abeta_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    % imagesc(DataForStat.pTau_DAB_ref{i}); colormap('jet'); % Plot the data
    imagesc(DataForStat.Abeta_DAB_Low{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat.MRI_Ref{i},[]); % Plot the data

    % Add colorbar to show value scale
    colorbar;
    clim([0 1]);

    axis off; % Remove axes
    label = DataForStat.cases{i}; 
    if label =='CN'
        title('CN', 'FontName', 'Arial'); % Add a title with Arial font
    elseif label=='AD'
        title('AD', 'FontName', 'Arial'); % Add a title with Arial font
    end
end

% Adjust layout for better visualization
% sgtitle('Hyperphosphorylated Tau', 'FontName', 'Arial'); % Add title
sgtitle('Amyloid Beta Low', 'FontName', 'Arial'); % Add title
% sgtitle('Reference MRI', 'FontName', 'Arial'); % Add title

% Save the plot
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'Amyloid_Beta_Low.png'), '-dpng', '-r600')

%% Amyloid Beta ref

% Width: 12 inches, Height: 3 inches
figure('Units', 'inches', 'Position', [1, 1, 24, 5]);

% Loop through the specified indices
% Abeta_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    % imagesc(DataForStat.pTau_DAB_ref{i}); colormap('jet'); % Plot the data
    imagesc(DataForStat.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat.MRI_Ref{i},[]); % Plot the data

    % Add colorbar to show value scale
    colorbar;
    clim([0 1]);

    axis off; % Remove axes
    label = DataForStat.cases{i}; 
    if label =='CN'
        title('CN', 'FontName', 'Arial'); % Add a title with Arial font
    elseif label=='AD'
        title('AD', 'FontName', 'Arial'); % Add a title with Arial font
    end
end

% Adjust layout for better visualization
% sgtitle('Hyperphosphorylated Tau', 'FontName', 'Arial'); % Add title
sgtitle('Amyloid Beta ref', 'FontName', 'Arial'); % Add title
% sgtitle('Reference MRI', 'FontName', 'Arial'); % Add title

% Save the plot
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'Amyloid_Beta_ref.png'), '-dpng', '-r600')

%% Plot all sample voxels in the image, T1D, as a figure panel

% To reconstruct the ith Psi(D,T1) as an image, use
% indices = [1, 3, 5, 7]; % Indices to process
indices = [1, 2, 3, 4, 5, 6, 7, 8];

% Create a single figure
figure('Units', 'inches', 'Position', [1, 1, 20, 5]); % Adjusted size for 4 images in a row

% Loop through specified indices
for idx = 1:length(indices)
    i = indices(idx); % Current index
    
    % Initialize the output array with the correct dimensions
    Im = zeros([size(DataForStat.PsiT1D{i}, 1), size(DataForStat.PsiT1D{i}, 2), size(DataForStat.bwMaskLowFinal{i})]);
    
    % Loop through each element in the row_pixCurFinal and col_pixCurFinal arrays
    for j = 1:length(DataForStat.row_pixCurFinal{i})
        % Assign the data to the correct location in the Im matrix
        row = DataForStat.row_pixCurFinal{i}(j);
        col = DataForStat.col_pixCurFinal{i}(j);
        Im(:,:,row, col) = squeeze(DataForStat.PsiT1D{i}(:,:,j));
    end

    size(Im)
    
    % Create a subplot for each image
    subplot(1, length(indices), idx); % 1 row, length(indices) columns
    contourf(Im(:,:,65,50)); % Plot the contour of the specific voxel
    
    % Label the headings according to the type of subject corresponding to that slice
    subject = DataForStat.cases{i};
    if subject == 'CN'
        title('CN'); % Add a title indicating the index
    elseif subject == 'AD'
        title('AD'); % Add a title indicating the index
    end
    ylabel('T_1 [ms]'); xlabel('MD [\mum^2/ms]'); 
    grid on; % Enable grid lines
    axis equal; % Ensure equal aspect ratio
end

sgtitle('T_1-MD Joint Distributions', 'FontName', 'Arial'); % Add title

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'PsiT1D.png'), '-dpng', '-r600')

%% Plot all sample voxels in the image, T2D, as a figure panel

% To reconstruct the ith Psi(D,T2) as an image, use
% indices = [1, 3, 5, 7]; % Indices to process
indices = [1, 2, 3, 4, 5, 6, 7, 8];

% Create a single figure
figure('Units', 'inches', 'Position', [1, 1, 20, 5]); % Adjusted size for 8 images in a row

% Loop through specified indices
for idx = 1:length(indices)
    i = indices(idx); % Current index
    
    % Initialize the output array with the correct dimensions
    % Im [50, 50, row, col]
    Im = zeros([size(DataForStat.PsiT2D{i}, 1), size(DataForStat.PsiT2D{i}, 2), size(DataForStat.bwMaskLowFinal{i})]);
    
    % Loop through each element in the row_pixCurFinal and col_pixCurFinal arrays
    for j = 1:length(DataForStat.row_pixCurFinal{i})
        % Assign the data to the correct location in the Im matrix
        row = DataForStat.row_pixCurFinal{i}(j);
        col = DataForStat.col_pixCurFinal{i}(j);
        Im(:,:,row, col) = squeeze(DataForStat.PsiT2D{i}(:,:,j));
    end
    
    % Create a subplot for each image
    subplot(1, length(indices), idx); % 1 row, length(indices) columns
    contourf(Im(:,:,65,50)); % Plot the contour of the specific voxel 64 row 50 col in 8 different slices
    
    % Label the headings according to the type of subject corresponding to that slice
    subject = DataForStat.cases{i};
    if subject == 'CN'
        title('CN'); % Add a title indicating the index
    elseif subject == 'AD'
        title('AD'); % Add a title indicating the index
    end
    ylabel('T_2 [ms]'); xlabel('MD [\mum^2/ms]'); 
    grid on; % Enable grid lines
    axis equal; % Ensure equal aspect ratio
end

sgtitle('T_2-MD Joint Distributions', 'FontName', 'Arial'); % Add title

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'PsiT2D.png'), '-dpng', '-r600')

%% Integral of examples of T1D and T2D

% store the integral of T1D and T2D
T1D_integrals = zeros(1,length(indices));
T2D_integrals = zeros(1,length(indices));

% calculate the integral for each slice
for i = 1:length(indices)
    % calculate the integrals of T1D and T2D for all voxels
    T1D_integrals(i) = mean(sum(sum(DataForStat.PsiT1D{i}, 1), 2), 'all');
    T2D_integrals(i) = mean(sum(sum(DataForStat.PsiT2D{i}, 1), 2), 'all');
end

% display the result of integral
disp('PsiT1D Integral result:');
disp(T1D_integrals);

disp('PsiT2D Integral result:');
disp(T2D_integrals);

%% Construct the dataset of pTau and Abeta

nSlices = numel(DataForStat.bwMaskLowFinal);

% Extract pTau values by voxel coordinates
pTau_all = []; % Store all voxel pTau values
slice_id = []; % Store slice IDs for each voxel

for i = 1:nSlices
    rows = DataForStat.row_pixCurFinal{i}; % Row coordinates of valid voxels
    cols = DataForStat.col_pixCurFinal{i}; % Column coordinates of valid voxels
    
    % Convert (row, col) to linear indices
    idx_lin = sub2ind(size(DataForStat.pTau_DAB_Low{i}), rows, cols);
    
    % Extract pTau values for these coordinates
    vals = DataForStat.pTau_DAB_Low{i}(idx_lin);

    % Append to the full dataset
    pTau_all = [pTau_all; vals(:)];
    slice_id = [slice_id; repmat(i, numel(vals), 1)];
end

% Save pTau dataset
save('dataset/pTau_data.mat', 'pTau_all', 'slice_id');
fprintf('pTau data saved: %d voxels\n', numel(pTau_all));

% Extract Abeta values by voxel coordinates
Abeta_all = []; % Store all voxel Abeta values
slice_id = []; % Store slice IDs for each voxel

for i = 1:nSlices
    rows = DataForStat.row_pixCurFinal{i};  % Row coordinates of valid voxels
    cols = DataForStat.col_pixCurFinal{i};  % Column coordinates of valid voxels
    
    % Convert (row, col) to linear indices
    idx_lin = sub2ind(size(DataForStat.Abeta_DAB_Low{i}), rows, cols);
    
    % Extract Abeta values for these coordinates
    vals = DataForStat.Abeta_DAB_Low{i}(idx_lin);

    % Append to the full dataset
    Abeta_all = [Abeta_all; vals(:)];
    slice_id = [slice_id; repmat(i, numel(vals), 1)];
end

% Save Abeta dataset
save('dataset/Abeta_data.mat', 'Abeta_all', 'slice_id');
fprintf('Abeta data saved: %d voxels\n', numel(Abeta_all));

%% Distribution Plot of value of pTau and Abeta

figure('Position',[100,100,1200,400]);

% pTau Distribution
subplot(1,2,1);
histogram(pTau_all, 50, 'FaceColor',[0.2 0.4 0.8], 'EdgeColor','none');
xlabel('pTau Intensity'); ylabel('Frequency');
title('pTau Distribution');
% grid on;

% Abeta Distribution
subplot(1,2,2);
histogram(Abeta_all, 50, 'FaceColor',[0.8 0.3 0.3], 'EdgeColor','none');
xlabel('Abeta Intensity'); ylabel('Frequency');
title('Abeta Distribution');
% grid on;

sgtitle('pTau and Abeta Distributions');

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'pTau_Abeta_Distribution.png'), '-dpng', '-r600')

% % pTau
% figure('Position',[200,200,600,450]);
% histogram(pTau_all, 50, 'FaceColor',[0.2 0.4 0.8], 'EdgeColor','none');
% xlabel('pTau intensity');
% ylabel('Voxel count');
% title('Distribution of pTau');
% set(gcf,'PaperPositionMode','auto');
% print(gcf, fullfile('image/dataset','pTau_distribution.png'), '-dpng','-r600');
% 
% % Abeta
% figure('Position',[200,200,600,450]);
% histogram(Abeta_all, 50, 'FaceColor',[0.8 0.3 0.3], 'EdgeColor','none');
% xlabel('Abeta intensity');
% ylabel('Voxel count');
% title('Distribution of Abeta');
% set(gcf,'PaperPositionMode','auto');
% print(gcf, fullfile('image/dataset','Abeta_distribution.png'), '-dpng','-r600');

%% Convert continuous Abeta data to binary / ternary labels using Otsu threshold

% Load your continuous Abeta data
load('dataset/Abeta_data.mat', 'Abeta_all', 'slice_id');  % Replace with your actual file name/variable name

% Binary classification
% Find a single threshold using Otsu's method
thresh_bin = multithresh(Abeta_all, 1);  % 1 threshold => 2 classes
fprintf('Binary Otsu threshold = %.4f\n', thresh_bin);

% Create binary labels: 0 = low Abeta, 1 = high Abeta
Abeta_binary = Abeta_all > thresh_bin;

% Ternary classification
% Find two thresholds using Otsu's method
thresh_tri = multithresh(Abeta_all, 2);  % 2 thresholds => 3 classes
fprintf('Ternary Otsu thresholds = %.4f, %.4f\n', thresh_tri(1), thresh_tri(2));

% Create ternary labels: 0 = low, 1 = medium, 2 = high
Abeta_ternary = imquantize(Abeta_all, thresh_tri) - 1;

% Save the results
save('dataset/Abeta_classified.mat', 'Abeta_all', 'Abeta_binary', 'Abeta_ternary', 'slice_id');

% Plot histograms with thresholds
figure('Position', [100, 100, 1000, 400]);

subplot(1,2,1);
histogram(Abeta_all, 50, 'FaceColor',[0.8 0.3 0.3], 'EdgeColor','none');
hold on;
xline(thresh_bin, 'b--', 'LineWidth', 2);
xlabel('Abeta Intensity');
ylabel('Frequency');
title('Binary Threshold (Otsu)');
grid on;

subplot(1,2,2);
histogram(Abeta_all, 50, 'FaceColor',[0.8 0.3 0.3], 'EdgeColor','none');
hold on;
xline(thresh_tri(1), 'b--', 'LineWidth', 2);
xline(thresh_tri(2), 'g--', 'LineWidth', 2);
xlabel('Abeta Intensity');
ylabel('Frequency');
title('Ternary Thresholds (Otsu)');
grid on;

sgtitle('Abeta Classification via Otsu Method');

% Save the plot
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'Abeta_Otsu_Classification.png'), '-dpng', '-r600')

%% Construct the dataset of T1D

indices = 1:8;

% Initialize the feature matrix X and target vector y
X = [];  % Each row corresponds to a voxel with 2500-dimensional features
y = [];  % Corresponding tau concentration in each voxel

% Iterate over all slices
for i = 1:length(indices)
    % Get the number of valid voxels in the i-th slice
    num_voxels = size(DataForStat.PsiT1D{i}, 3);
    % Iterate over each valid voxel in the slice
    for j = 1:num_voxels
        % Extract the 50×50 pdf of the voxel
        pdf_matrix = DataForStat.PsiT1D{i}(:,:,j);

        % Flatten it into a 2500-dimensional row vector
        pdf_vector = pdf_matrix(:)';

        % Append the vector to the feature matrix
        X = [X; pdf_vector];

        % The target value is the tau_mean of the slice (same for all voxels in the slice)
        % y = [y; tau_mean(i)];

        % The target value is the concentration of pTau in each voxel
        % Get (row, col) of current voxel
        row = DataForStat.row_pixCurFinal{i}(j);
        col = DataForStat.col_pixCurFinal{i}(j);

        % Get the voxel-level pTau concentration
        ptau_value = DataForStat.pTau_DAB_Low{i}(row, col);

        % Append to y
        y = [y; ptau_value];
    end
end

% Display the dimensions of the constructed data
fprintf('X size: %d x %d\n', size(X,1), size(X,2));
fprintf('y size: %d x %d\n', size(y,1), size(y,2));

% Store X and y from T1D
save('dataset/regression_data_T1D.mat', 'X', 'y');

%% Construct the dataset of T2D

indices = 1:8;

% Initialize the feature matrix X and target vector y
X = [];  % Each row corresponds to a voxel with 2500-dimensional features
y = [];  % Corresponding tau concentration in each voxel

% Iterate over all slices
for i = 1:length(indices)
    % Get the number of valid voxels in the i-th slice
    num_voxels = size(DataForStat.PsiT2D{i}, 3);
    % Iterate over each valid voxel in the slice
    for j = 1:num_voxels
        % Extract the 50×50 pdf of the voxel
        pdf_matrix = DataForStat.PsiT2D{i}(:,:,j);

        % Flatten it into a 2500-dimensional row vector
        pdf_vector = pdf_matrix(:)';

        % Append the vector to the feature matrix
        X = [X; pdf_vector];

        % The target value is the tau_mean of the slice (same for all voxels in the slice)
        % y = [y; tau_mean(i)];

        % The target value is the concentration of pTau in each voxel
        % Get (row, col) of current voxel
        row = DataForStat.row_pixCurFinal{i}(j);
        col = DataForStat.col_pixCurFinal{i}(j);

        % Get the voxel-level pTau concentration
        ptau_value = DataForStat.pTau_DAB_Low{i}(row, col);

        % Append to y
        y = [y; ptau_value];
    end
end

% Display the dimensions of the constructed data
fprintf('X size: %d x %d\n', size(X,1), size(X,2));
fprintf('y size: %d x %d\n', size(y,1), size(y,2));

% Store X and y from T2D
save('dataset/regression_data_T2D.mat', 'X', 'y');

%% Construct the dataset for image regression of T1D

indices = 1:8;

% 1. Compute total number of voxels across all slices
total_voxels = 0;
for i = 1:length(indices)
    total_voxels = total_voxels + size(DataForStat.PsiT1D{i}, 3);
end

% 2. Preallocate CNN input tensor X and label vector y
% X will be [50 × 50 × 1 × total_voxels], i.e. height×width×channels×batch
X = zeros(50, 50, 1, total_voxels, 'single');
% y will be [total_voxels × 1], each entry is the pTau concentration
y = zeros(total_voxels, 1, 'single');

% 3. Fill X and y with voxel-wise data
cnt = 0;
for i = 1:length(indices)
    num_voxels = size(DataForStat.PsiT1D{i}, 3);
    for j = 1:num_voxels
        cnt = cnt + 1;

        % 3.1 Extract the 50×50 probability distribution for voxel j
        pdf_matrix = DataForStat.PsiT1D{i}(:,:,j);   % [50×50]

        % 3.2 Assign it to the CNN input tensor
        X(:,:,1,cnt) = pdf_matrix;

        % 3.3 Look up the corresponding pTau concentration
        row = DataForStat.row_pixCurFinal{i}(j);
        col = DataForStat.col_pixCurFinal{i}(j);
        y(cnt) = DataForStat.pTau_DAB_Low{i}(row, col);
    end
end

% 4. Verify tensor dimensions
fprintf('X size: %d × %d × %d × %d\n', size(X,1), size(X,2), size(X,3), size(X,4));
fprintf('y size: %d × %d\n', size(y,1), size(y,2));

% 5. Save the dataset for later use
save('dataset/regression_image_data_T1D.mat', 'X', 'y', '-v7.3');

%% Construct the dataset for image regression of T2D

indices = 1:8;

% 1. Compute total number of voxels across all slices
total_voxels = 0;
for i = 1:length(indices)
    total_voxels = total_voxels + size(DataForStat.PsiT2D{i}, 3);
end

% 2. Preallocate CNN input tensor X and label vector y
%    X will be [50 × 50 × 1 × total_voxels]: height × width × channels × batch
X = zeros(50, 50, 1, total_voxels, 'single');
%    y will be [total_voxels × 1], each entry is the pTau concentration
y = zeros(total_voxels, 1, 'single');

% 3. Fill X and y with voxel-wise data
cnt = 0;
for i = 1:length(indices)
    num_voxels = size(DataForStat.PsiT2D{i}, 3);
    for j = 1:num_voxels
        cnt = cnt + 1;

        % 3.1 Extract the 50×50 probability distribution for voxel j (T2D)
        pdf_matrix = DataForStat.PsiT2D{i}(:,:,j);   % [50×50]

        % 3.2 Assign it to the CNN input tensor
        X(:,:,1,cnt) = pdf_matrix;

        % 3.3 Look up the corresponding pTau concentration
        row = DataForStat.row_pixCurFinal{i}(j);
        col = DataForStat.col_pixCurFinal{i}(j);
        y(cnt) = DataForStat.pTau_DAB_Low{i}(row, col);
    end
end

% 4. Verify tensor dimensions
fprintf('X size: %d × %d × %d × %d\n', size(X,1), size(X,2), size(X,3), size(X,4));
fprintf('y size: %d × %d\n', size(y,1), size(y,2));

% 5. Save the dataset for later use
save('dataset/regression_image_data_T2D.mat', 'X', 'y', '-v7.3');

%% Binary dataset creation based on Otsu method

% Transfer the following 4 datasets into binary version
fileList = {
    'dataset/regression_data_T1D.mat', ...
    'dataset/regression_data_T2D.mat', ...
    'dataset/regression_image_data_T1D.mat', ...
    'dataset/regression_image_data_T2D.mat'
};

% Take one dataset as an example, since they have the same target value
S = load(fileList{1}, 'y');
y_all = S.y;

thresh = multithresh(y_all, 1);   % single Otsu threshold
fprintf('Computed Otsu threshold = %.4f\n', thresh);

% Visualize the y histogram with the threshold
figure;
histogram(y_all, 50);
hold on;
xline(thresh, 'r--', 'LineWidth', 2);
title('Histogram of y with Otsu threshold');
xlabel('y');
ylabel('Frequency');
legend('y values', 'Otsu threshold');
grid on;
exportgraphics(gcf, fullfile('image/dataset','histogram_y_otsu_binary.png'), 'Resolution', 300);

% Loop over each dataset to binarize and save
for k = 1:numel(fileList)
    thisFile = fileList{k};
    S = load(thisFile, 'X', 'y');
    X = S.X;
    y = S.y;

    % Binarize using the precomputed threshold
    yBinary = y > thresh;

    % Generate output filename: change "regression" to "classification"
    [pathstr, name, ~] = fileparts(thisFile);
    newName = strrep(name, 'regression', 'classification');
    outFile = fullfile(pathstr, [newName '_binary.mat']);

    % Save binary dataset
    save(outFile, 'X', 'yBinary', 'thresh', '-v7.3');
    fprintf('Saved binary dataset to "%s"\n', outFile);
end

%% Visualization of y

% Load the data of T1D as an example
load('dataset/regression_data_T1D.mat');

% Display dataset dimensions
fprintf('Dataset dimensions:\n');
fprintf('X: %d × %d × %d × %d\n', size(X));  
fprintf('y: %d × %d\n\n', size(y));

% Visualization of y
figure;
plot(1:numel(y), y, '.', 'MarkerSize', 8);
xlim([0,numel(y)])
ylim([0,1])
% title('Scatter of y by Sample Index');
xlabel('Sample Index');
ylabel('pTau Concentration');

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'y_distribution.png'), '-dpng', '-r600')

%% Visualization of binary y

% Load the data of T1D as an example
load('dataset/classification_data_T1D_binary.mat');

% disp(yBinary)

% Display dataset dimensions
fprintf('Dataset dimensions:\n');
fprintf('X: %d × %d × %d × %d\n', size(X));  
fprintf('y: %d × %d\n\n', size(yBinary));

% Visualization of y
figure;
plot(1:numel(yBinary), yBinary, '.', 'MarkerSize', 8);
xlim([0,numel(yBinary)])
ylim([0,1])
% title('Scatter of y by Sample Index');
xlabel('Sample Index');
ylabel('pTau Concentration');

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'y_distribution_binary.png'), '-dpng', '-r600')

%% Complete binary classification reconstruction and plotting script

% 1. Load DataForStat
load('dataset/DataForStat_101124.mat');

% 2. Load binary classification dataset
%   X        [N×2500]  % (unused)
%   yBinary  [N×1]     % 0/1 label for each valid voxel
load('dataset/classification_data_T1D_binary.mat', 'yBinary');

% 3. Compute start and end indices for each slice in yBinary vector
nSlices = numel(DataForStat.bwMaskLowFinal);
sliceStarts = zeros(nSlices, 1);
sliceEnds   = zeros(nSlices, 1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

% 4. Reconstruct each slice as a 125×89 binary image (ROI filled with 0/1, background is 0)
result = cell(1, nSlices);
for i = 1:nSlices
    mask_i = DataForStat.bwMaskLowFinal{i};
    img_i = zeros(size(mask_i)); % initialize background to 0
    labels = yBinary(sliceStarts(i):sliceEnds(i));
    rows   = DataForStat.row_pixCurFinal{i};
    cols   = DataForStat.col_pixCurFinal{i};
    for j = 1:numel(labels)
        img_i(rows(j), cols(j)) = labels(j);
    end
    result{i} = img_i;
end

% 5. Display the binary classification results side by side
indices = 1:nSlices; % [1 2 3 4 5 6 7 8]

figure('Units','inches','Position',[1 1 24 5]);
for idx = 1:length(indices)
    i = indices(idx);
    subplot(1, nSlices, idx);
    
    imagesc(result{i});    % show binary image
    colormap('jet');       % map 0->blue, 1->red
    colorbar;
    clim([0 1]);           % enforce mapping range [0,1]
    axis off;        % hide axes and maintain aspect ratio
    
    % Get pixel dimensions
    [h, w] = size(result{i});
    pixText = sprintf('%d × %d', h, w);
    
    % Add title based on case label and pixel info
    if DataForStat.cases{i} == 'CN'
        title({'CN'}, 'FontName', 'Arial');
    else
        title({'AD'}, 'FontName', 'Arial');
    end
end

% 6. Global title
sgtitle('Binary Classified pTau', 'FontName', 'Arial');

% 7. Export high-resolution figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'pTau_binary_classified.png'), '-dpng', '-r600')

%% Complete regression reconstruction and plotting script

% 1. Load DataForStat
load('dataset/DataForStat_101124.mat');

% 2. Load regression dataset
load('dataset/regression_data_T1D.mat', 'y');

% 3. Compute start and end indices for each slice in y vector
nSlices = numel(DataForStat.bwMaskLowFinal);
sliceStarts = zeros(nSlices, 1);
sliceEnds   = zeros(nSlices, 1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

% 4. Reconstruct each slice as a 125×89 image (ROI filled with 0/1, background is 0)
result = cell(1, nSlices);
for i = 1:nSlices
    mask_i = DataForStat.bwMaskLowFinal{i};
    img_i = zeros(size(mask_i)); % initialize background to 0
    labels = y(sliceStarts(i):sliceEnds(i));
    rows   = DataForStat.row_pixCurFinal{i};
    cols   = DataForStat.col_pixCurFinal{i};
    for j = 1:numel(labels)
        img_i(rows(j), cols(j)) = labels(j);
    end
    result{i} = img_i;
end

% 5. Display the binary classification results side by side
indices = 1:nSlices; % [1 2 3 4 5 6 7 8]

figure('Units','inches','Position',[1 1 24 5]);
for idx = 1:length(indices)
    i = indices(idx);
    subplot(1, nSlices, idx);
    
    imagesc(result{i});    % show binary image
    colormap('jet');       % map 0->blue, 1->red
    colorbar;
    clim([0 1]);           % enforce mapping range [0,1]
    axis off;        % hide axes and maintain aspect ratio
    
    % Get pixel dimensions
    [h, w] = size(result{i});
    pixText = sprintf('%d × %d', h, w);
    
    % Add title based on case label
    if DataForStat.cases{i} == 'CN'
        title({'CN'}, 'FontName', 'Arial');
    else
        title({'AD'}, 'FontName', 'Arial');
    end
end

% 6. Global title
sgtitle('Continuous pTau', 'FontName', 'Arial');

% 7. Export high-resolution figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'pTau_continuous.png'), '-dpng', '-r600')

%% Three‐Class Dataset Creation with Class‐Mean Target Values (Using Otsu Thresholding)

% List of the four datasets to process
fileList = {
    'dataset/regression_data_T1D.mat', ...
    'dataset/regression_data_T2D.mat', ...
    'dataset/regression_image_data_T1D.mat', ...
    'dataset/regression_image_data_T2D.mat'
};

% 1. Load the first dataset and extract its continuous labels 'y'
S = load(fileList{1}, 'y');
y_all = S.y;  
%    y_all is a vector of length N containing the original pTau concentration values

% 2. Compute two optimal Otsu thresholds to split y_all into three groups
thresholds = multithresh(y_all, 2);   % returns [t1, t2]
fprintf('Computed Otsu thresholds = [%.4f, %.4f]\n', thresholds);

% 3. Plot the histogram of y_all with the two threshold lines
figure;
histogram(y_all, 50);
hold on;
xline(thresholds(1), 'r--', 'LineWidth', 2);
xline(thresholds(2), 'b--', 'LineWidth', 2);
title('Histogram of y with Two Otsu Thresholds');
xlabel('y');
ylabel('Frequency');
legend('y values', 'Otsu threshold 1', 'Otsu threshold 2');
grid on;
exportgraphics(gcf, fullfile('image/dataset', 'histogram_y_otsu_ternary.png'), 'Resolution', 300);

% 4. Loop over each dataset, create a new continuous target equal to the mean
%    pTau of its Otsu‐defined group, and save
for k = 1:numel(fileList)
    thisFile = fileList{k};
    
    % 4a. Load feature matrix X (unused here) and original labels y
    S = load(thisFile, 'X', 'y');
    X = S.X;  
    y = S.y;  

    % 4b. Discretize y into three groups using the precomputed thresholds:
    %     Group 1: y < t1, Group 2: t1 <= y < t2, Group 3: y >= t2
    edges = [-Inf; thresholds(:); Inf];
    yGroup = discretize(y, edges, 1:3);

    % 4c. Compute the mean of original y for each group
    classMeans = arrayfun(@(c) mean(y(yGroup == c)), 1:3);

    % 4d. Replace each sample’s label by its group’s mean pTau
    %     Now yClassMean is a continuous target vector (same size as y)
    yClassMean = classMeans(yGroup)';

    % 4e. Generate the output filename by replacing 'regression' with 'classification'
    [pathstr, name, ~] = fileparts(thisFile);
    newName = strrep(name, 'regression', 'classification');
    outFile = fullfile(pathstr, [newName '_ternary.mat']);

    % 4f. Save the feature matrix and the new continuous target
    save(outFile, 'X', 'yClassMean', 'thresholds', '-v7.3');
    fprintf('Saved class‐mean target dataset to "%s"\n', outFile);
end

%% Distribution of ternary y

% Load the data of T1D as an example
load('dataset/classification_data_T1D_ternary.mat');

% disp(yClassMean)

% Display dataset dimensions
fprintf('Dataset dimensions:\n');
fprintf('X: %d × %d × %d × %d\n', size(X));  
fprintf('y: %d × %d\n\n', size(yClassMean));

% Visualization of y
figure;
plot(1:numel(yClassMean), yClassMean, '.', 'MarkerSize', 8);
xlim([0,numel(yClassMean)])
ylim([0,1])
% title('Scatter of y by Sample Index');
xlabel('Sample Index');
ylabel('pTau Concentration');

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'y_distribution_ternary.png'), '-dpng', '-r600')

%% Complete ternary classification reconstruction and plotting script

% 1. Load DataForStat
load('dataset/DataForStat_101124.mat');

% 2. Load binary classification dataset
%   X  [N×2500]
%   yClassMean  [N×1]  average value of each class
load('dataset/classification_data_T1D_ternary.mat');

% 3. Compute start and end indices for each slice in yClassMean vector
nSlices = numel(DataForStat.bwMaskLowFinal);
sliceStarts = zeros(nSlices, 1);
sliceEnds   = zeros(nSlices, 1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

% 4. Reconstruct each slice as a 125×89 binary image (ROI filled with 0/1, background is 0)
result = cell(1, nSlices);
for i = 1:nSlices
    mask_i = DataForStat.bwMaskLowFinal{i};
    img_i = zeros(size(mask_i)); % initialize background to 0
    labels = yClassMean(sliceStarts(i):sliceEnds(i));
    rows   = DataForStat.row_pixCurFinal{i};
    cols   = DataForStat.col_pixCurFinal{i};
    for j = 1:numel(labels)
        img_i(rows(j), cols(j)) = labels(j);
    end
    result{i} = img_i;
end

% 5. Display the binary classification results side by side
indices = 1:nSlices;

figure('Units','inches','Position',[1 1 24 5]);
for idx = 1:length(indices)
    i = indices(idx);
    subplot(1, nSlices, idx);
    
    imagesc(result{i});    % show binary image
    colormap('jet');       % map 0->blue, 1->red
    colorbar;
    clim([0 1]);           % enforce mapping range [0,1]
    axis off;        % hide axes and maintain aspect ratio
    
    % Get pixel dimensions
    [h, w] = size(result{i});
    pixText = sprintf('%d × %d', h, w);
    
    % Add title based on case label
    if DataForStat.cases{i} == 'CN'
        title({'CN'}, 'FontName', 'Arial');
    else
        title({'AD'}, 'FontName', 'Arial');
    end
end

% 6. Global title
sgtitle('Ternary Classified pTau', 'FontName', 'Arial');

% 7. Export high-resolution figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/dataset', 'pTau_ternary_classified.png'), '-dpng', '-r600')


