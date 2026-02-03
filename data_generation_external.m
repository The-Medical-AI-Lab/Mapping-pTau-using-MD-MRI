
clc; clear; close all;

%% Load the data

load("external dataset/DataForStat_120825.mat");

%% Standardize labels in DataForStat2.cases from Cont to CN

c = DataForStat2.cases;
wasCell = iscell(c);

cs = string(c); % normalize to string for safe editing
cs = strtrim(cs); % remove leading/trailing spaces
mask = strcmpi(cs,"cont"); % case-insensitive match for 'cont'
nRepl = nnz(mask);
cs(mask) = "CN"; % replace with CN

% Restore original type
if wasCell
    DataForStat2.cases = cellstr(cs);
else
    DataForStat2.cases = cs;
end

%% To reconstruct the ith Psi(D,T1) as an image

i = 4; % range from 1 to 14 (14 slices), choose the 4 as an example

% Voxel means Volume Pixel (valid voxel and invalid or background voxel)
% Initialize the output array with the correct dimensions
% Im is initialized as an all zero matrix, whose size is based on the size of PsiT1D data and bwMaskLowFinal
% Im is 4 dimensional, 50x50x125x89
% Dimension 1 and 2: Same as the first two dimensions of PsiT1D{i} (50×50), representing the spatial dimension of the localized image.
% Dimensions 3 and 4: Same size as bwMaskLowFinal{i} (125×89), representing the spatial coordinates of the original MRI image.

Im = zeros([size(DataForStat2.PsiT1D{i}, 1), size(DataForStat2.PsiT1D{i}, 2), size(DataForStat2.bwMaskLowFinal{i})]);

disp(size(Im(:,:,:,:))) % 50 50 128 107
disp(size(DataForStat2.PsiT1D{i})) % 50 50 9632
disp(size(DataForStat2.PsiT1D{i})) % 50 50 9632
disp(size(DataForStat2.pTau_DAB_Low{i})) % 128 107
disp(size(DataForStat2.row_pixCurFinal{i})) % 9632 1
disp(size(DataForStat2.col_pixCurFinal{i})) % 9632 1
disp(size(DataForStat2.bwMaskLowFinal{i})) % 128 107
disp(sum(DataForStat2.bwMaskLowFinal{i}(:))); % 9632 ROI

% So the sum of all 1 in bwMaskLowFinal is valid voxel
% White (value of 1) represent the region of interest (ROI)
% Black (value of 0) represent a background or invalid area

% Loop through each element in the row_pixCurFinal and col_pixCurFinal arrays
% According to the coordinates given by row_pixCurFinal {i} and col_pixCurFinal {i}
% map the 50 × 50 dimensional data corresponding to PsiT1D to the (row, col) position of the MRI image
for j = 1:length(DataForStat2.row_pixCurFinal{i})
    % Traverse the positions of all valid voxels in the ROI
    row = DataForStat2.row_pixCurFinal{i}(j);
    col = DataForStat2.col_pixCurFinal{i}(j);
    % Extract the jth 50x50 segment of the PsiT1D{i}
    Im(:,:,row, col) = squeeze(DataForStat2.PsiT1D{i}(:,:,j));
end

%% Visaulize ROI (region of interest) for each slice, and we only use vaild voxels to construct new dataset (bwMaskLow)

figure('Name','All ROI Masks','NumberTitle','off','Units', 'inches', 'Position', [1, 1, 16, 5]);
valid_amount = 0;
valid_num = [];

for i = 1:6
    % bwMaskLow: initial binary mask from low‐threshold segmentation
    % bwMaskLowFinal: final ROI mask after morphological cleaning and connected‐component filtering
    mask = DataForStat2.bwMaskLow{i};
    [H, W] = size(mask);
    validPixelCount = sum(mask(:));
    valid_amount = valid_amount + validPixelCount;
    % disp(validPixelCount);
    valid_num = [valid_num, validPixelCount];

    subplot(1,6,i);
    imagesc(mask);
    colormap(gray);
    % axis image off;
    title(sprintf('%d x %d \n %d valid voxels', H, W, validPixelCount), 'FontSize', 12);
end

sgtitle('Initial ROI Mask', 'FontName', 'Arial'); % Add title

% disp(valid_amount)

% Display the array of valid voxel counts
% disp('Valid voxel counts per slice:');
% disp(valid_num);
% disp(sum(valid_num));

% Plot the Binary Mask of each slice
set(gcf,'PaperPositionMode','auto');
print(gcf, fullfile('image/external dataset', 'ROI_Masks_bwMaskLow_external.png'), '-dpng', '-r600');

%% Visaulize ROI (region of interest) for each slice, and we only use vaild voxels to construct new dataset (bwMaskLowFinal)

figure('Name','All ROI Masks','NumberTitle','off','Units', 'inches', 'Position', [1, 1, 16, 5]);
valid_amount = 0;
valid_num = [];

for i = 1:6
    % bwMaskLow: initial binary mask from low‐threshold segmentation
    % bwMaskLowFinal: final ROI mask after morphological cleaning and connected‐component filtering
    mask = DataForStat2.bwMaskLowFinal{i};
    [H, W] = size(mask);
    validPixelCount = sum(mask(:));
    valid_amount = valid_amount + validPixelCount;
    % disp(validPixelCount);
    valid_num = [valid_num, validPixelCount];

    subplot(1,6,i);
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
set(gcf,'PaperPositionMode','auto');
print(gcf, fullfile('image/external dataset', 'ROI_Masks_bwMaskLowFinal_external.png'), '-dpng', '-r600');

%% Plot contours of joint distribution of some sample valid voxels

% figure('Units','inches','Position',[1 1 12 12]);  % [x, y, width, height]
% plotCount = 1;
% for row = 1:size(Im, 3)
%     for col = 1:size(Im, 4)
%         if nnz(Im(:,:,row,col)) > 0
%             if plotCount > 25
%                 break;
%             end
%             subplot(5,5,plotCount);
%             contourf(Im(:,:,row,col)); % Draw contour lines
%             title(['Row: ', num2str(row), ', Col: ', num2str(col)]);
%             plotCount = plotCount + 1;
%         end
%     end
% end

%% MRI Ref

% This is to create a figure panel using the histology stained images

% MATLAB script to plot pTau_DAB_ref for i side by side
% Indices to plot
indices = 1:6;

% Create a new figure
% Width: 24 inches, Height: 6 inches
figure('Units', 'inches', 'Position', [1, 1, 16, 5]);

% Loop through the specified indices
% pTau_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    % imagesc(DataForStat2.pTau_DAB_Low{i}); colormap('jet'); % Plot the data
    % imagesc(DataForStat2.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat2.MRI_Ref{i}, []); % Plot the data, display the i-th MRI image, this is imshow not imagesc
    imagesc(DataForStat2.MRI_Ref{i}); colormap('gray'); % This is imagesc not imshow

    % Add colorbar to show value scale
    colorbar;
    % clim([0 1]);

    axis off; % Remove axes
    label = DataForStat2.cases{i}; 
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
print(gcf, fullfile('image/external dataset', 'MRI_Ref_external.png'), '-dpng', '-r600');

%% Hyperphosphorylated Tau Low

% This is to create a figure panel using the histology stained images

% MATLAB script to plot pTau_DAB_ref for i side by side
% Indices to plot
indices = 1:6;

% Create a new figure
% Width: 36 inches, Height: 6 inches
figure('Units', 'inches', 'Position', [1, 1, 16, 5]);

% Loop through the specified indices
% pTau_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    imagesc(DataForStat2.pTau_DAB_Low{i}); colormap('jet'); % Plot the data
    % imagesc(DataForStat2.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat2.MRI_Ref{i},[]); % Plot the data, display the i-th MRI image

    % Add colorbar to show value scale
    colorbar;
    clim([0 1]);

    axis off; % Remove axes
    label = DataForStat2.cases{i}; 
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
print(gcf, fullfile('image/external dataset', 'Hyperphosphorylated_Tau_Low_external.png'), '-dpng', '-r600');

%% Hyperphosphorylated Tau ref

% This is to create a figure panel using the histology stained images

% MATLAB script to plot pTau_DAB_ref for i side by side
% Indices to plot
indices = 1:6;

% Create a new figure
% Width: 24 inches, Height: 5 inches
figure('Units', 'inches', 'Position', [1, 1, 16, 5]);

% Loop through the specified indices
% pTau_DAB_ref
for idx = 1:length(indices)
    i = indices(idx); % Get the current index
    subplot(1, length(indices), idx); % Create a subplot for each figure
    imagesc(DataForStat2.pTau_DAB_ref{i}); colormap('jet'); % Plot the data
    % imagesc(DataForStat2.Abeta_DAB_ref{i}); colormap('jet'); % Plot the data
    % imshow(DataForStat2.MRI_Ref{i},[]); % Plot the data, display the i-th MRI image

    % Add colorbar to show value scale
    colorbar;
    clim([0 1]);

    axis off; % Remove axes
    label = DataForStat2.cases{i}; 
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
print(gcf, fullfile('image/external dataset', 'Hyperphosphorylated_Tau_ref_external.png'), '-dpng', '-r600')

%% Plot all sample voxels in the image, T1D, as a figure panel

% To reconstruct the ith Psi(D,T1) as an image
indices = 1:6;

% Create a single figure
figure('Units', 'inches', 'Position', [1, 1, 15, 5]); % Adjusted size for 4 images in a row

% Loop through specified indices
for idx = 1:length(indices)
    i = indices(idx); % Current index
    
    % Initialize the output array with the correct dimensions
    Im = zeros([size(DataForStat2.PsiT1D{i}, 1), size(DataForStat2.PsiT1D{i}, 2), size(DataForStat2.bwMaskLowFinal{i})]);
    
    % Loop through each element in the row_pixCurFinal and col_pixCurFinal arrays
    for j = 1:length(DataForStat2.row_pixCurFinal{i})
        % Assign the data to the correct location in the Im matrix
        row = DataForStat2.row_pixCurFinal{i}(j);
        col = DataForStat2.col_pixCurFinal{i}(j);
        Im(:,:,row, col) = squeeze(DataForStat2.PsiT1D{i}(:,:,j));
    end

    size(Im)
    
    % Create a subplot for each image
    subplot(1, length(indices), idx); % 1 row, length(indices) columns
    contourf(Im(:,:,85,50)); % Plot the contour of the specific voxel
    
    % Label the headings according to the type of subject corresponding to that slice
    subject = DataForStat2.cases{i};
    if subject == 'CN'
        title(['CN']); % Add a title indicating the index
    elseif subject == 'AD'
        title(['AD']); % Add a title indicating the index
    end
    ylabel('T_1 [ms]'); xlabel('MD [\mum^2/ms]'); 
    grid on; % Enable grid lines
    axis equal; % Ensure equal aspect ratio
end

sgtitle('T_1-MD Joint Distributions', 'FontName', 'Arial'); % Add title

set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/external dataset', 'PsiT1D_external.png'), '-dpng', '-r600')

%% Integral of examples of T1D

% store the integral of T1D
T1D_integrals = zeros(1,length(indices));

% calculate the integral for each slice
for i = 1:length(indices)
    % calculate the integrals of T1D and T2D for all voxels
    T1D_integrals(i) = mean(sum(sum(DataForStat2.PsiT1D{i}, 1), 2), 'all');
end

% display the result of integral
disp('PsiT1D Integral result:');
disp(T1D_integrals);

%% Construct the dataset of pTau and Abeta

nSlices = numel(DataForStat2.bwMaskLowFinal);

% Extract pTau values by voxel coordinates
pTau_all = [];       % Store all voxel pTau values
slice_id = [];       % Store slice IDs for each voxel

for i = 1:nSlices
    rows = DataForStat2.row_pixCurFinal{i};  % Row coordinates of valid voxels
    cols = DataForStat2.col_pixCurFinal{i};  % Column coordinates of valid voxels
    
    % Convert (row, col) to linear indices
    idx_lin = sub2ind(size(DataForStat2.pTau_DAB_Low{i}), rows, cols);
    
    % Extract pTau values for these coordinates
    vals = DataForStat2.pTau_DAB_Low{i}(idx_lin);

    % Append to the full dataset
    pTau_all = [pTau_all; vals(:)];
    slice_id = [slice_id; repmat(i, numel(vals), 1)];
end

% Save pTau dataset
save('external dataset/pTau_data_external.mat', 'pTau_all', 'slice_id');
fprintf('pTau data saved: %d voxels\n', numel(pTau_all));

%% Distribution Plot of value of pTau and Abeta

figure('Position',[100,100,600,400]);

% pTau Distribution
subplot(1,1,1);
histogram(pTau_all, 50, 'FaceColor',[0.2 0.4 0.8], 'EdgeColor','none');
xlabel('pTau Intensity'); ylabel('Frequency');
title('pTau Distribution');
% grid on;

sgtitle('pTau Distribution');

set(gcf,'PaperPositionMode','auto');
print(gcf, fullfile('image/external dataset','pTau_Distribution_external.png'), '-dpng','-r600');

%% Construct the dataset of T1D

indices = 1:6;

% Initialize the feature matrix X and target vector y
X = [];  % Each row corresponds to a voxel with 2500-dimensional features
y = [];  % Corresponding tau concentration in each voxel

% Iterate over all slices
for i = 1:length(indices)
    % Get the number of valid voxels in the i-th slice
    num_voxels = size(DataForStat2.PsiT1D{i}, 3);
    % Iterate over each valid voxel in the slice
    for j = 1:num_voxels
        % Extract the 50×50 pdf of the voxel
        pdf_matrix = DataForStat2.PsiT1D{i}(:,:,j);
        % Flatten it into a 2500-dimensional row vector
        pdf_vector = pdf_matrix(:)';
        % Append the vector to the feature matrix
        X = [X; pdf_vector];

        % The target value is the tau_mean of the slice (same for all voxels in the slice)
        % y = [y; tau_mean(i)];

        % The target value is the concentration of pTau in each voxel
        % Get (row, col) of current voxel
        row = DataForStat2.row_pixCurFinal{i}(j);
        col = DataForStat2.col_pixCurFinal{i}(j);

        % Get the voxel-level pTau concentration
        ptau_value = DataForStat2.pTau_DAB_Low{i}(row, col);

        % Append to y
        y = [y; ptau_value];
    end
end

% Display the dimensions of the constructed data
fprintf('X size: %d x %d\n', size(X,1), size(X,2));
fprintf('y size: %d x %d\n', size(y,1), size(y,2));

% Store X and y from T1D
save('external dataset/regression_data_T1D_external.mat', 'X', 'y', '-v7.3');

%% Construct the dataset for image regression of T1D

indices = 1:6;

% 1. Compute total number of voxels across all slices
total_voxels = 0;
for i = 1:length(indices)
    total_voxels = total_voxels + size(DataForStat2.PsiT1D{i}, 3);
end

% 2. Preallocate CNN input tensor X and label vector y
% X will be [50 × 50 × 1 × total_voxels], i.e. height×width×channels×batch
X = zeros(50, 50, 1, total_voxels, 'single');
% y will be [total_voxels × 1], each entry is the pTau concentration
y = zeros(total_voxels, 1, 'single');

% 3. Fill X and y with voxel-wise data
cnt = 0;
for i = 1:length(indices)
    num_voxels = size(DataForStat2.PsiT1D{i}, 3);
    for j = 1:num_voxels
        cnt = cnt + 1;

        % 3.1 Extract the 50×50 probability distribution for voxel j
        pdf_matrix = DataForStat2.PsiT1D{i}(:,:,j);   % [50×50]

        % 3.2 Assign it to the CNN input tensor
        X(:,:,1,cnt) = pdf_matrix;

        % 3.3 Look up the corresponding pTau concentration
        row = DataForStat2.row_pixCurFinal{i}(j);
        col = DataForStat2.col_pixCurFinal{i}(j);
        y(cnt) = DataForStat2.pTau_DAB_Low{i}(row, col);
    end
end

% 4. Verify tensor dimensions
fprintf('X size: %d × %d × %d × %d\n', size(X,1), size(X,2), size(X,3), size(X,4));
fprintf('y size: %d × %d\n', size(y,1), size(y,2));

% 5. Save the dataset for later use
save('external dataset/regression_image_data_T1D_external.mat', 'X', 'y', '-v7.3');

%% Binary dataset creation based on Otsu method

% Transfer the following 4 datasets into binary version
fileList = {'external dataset/regression_data_T1D_external.mat', 'external dataset/regression_image_data_T1D_external.mat'};

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
exportgraphics(gcf, fullfile('image/dataset','histogram_y_otsu_binary_external.png'), 'Resolution', 300);

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
    if endsWith(newName, '_external')
        newName = extractBefore(newName, strlength(newName) - strlength('_external') + 1);
    end
    outFile = fullfile(pathstr, [newName '_binary_external.mat']);

    % Save binary dataset
    save(outFile, 'X', 'yBinary', 'thresh', '-v7.3');
    fprintf('Saved binary dataset to "%s"\n', outFile);
end

%% Visualization of y

% Load the data of T1D as an example
load('external dataset/regression_data_T1D_external.mat');

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
print(gcf, fullfile('image/external dataset', 'y_distribution_external.png'), '-dpng', '-r600')

%% Visualization of binary y

% Load the data of T1D as an example
load('external dataset/classification_data_T1D_binary_external.mat');

disp(yBinary)

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
print(gcf, fullfile('image/external dataset', 'y_distribution_binary_external.png'), '-dpng', '-r600')

%% Complete binary classification reconstruction and plotting script

% 1. Load DataForStat2
% load("external dataset/DataForStat_120825.mat");

% 2. Load binary classification dataset
%   X        [N×2500]  % (unused)
%   yBinary  [N×1]     % 0/1 label for each valid voxel
load('external dataset/classification_data_T1D_binary_external.mat', 'yBinary');

% 3. Compute start and end indices for each slice in yBinary vector
nSlices = numel(DataForStat2.bwMaskLowFinal);
sliceStarts = zeros(nSlices, 1);
sliceEnds   = zeros(nSlices, 1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat2.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

% 4. Reconstruct each slice as a 125×89 binary image (ROI filled with 0/1, background is 0)
result = cell(1, nSlices);
for i = 1:nSlices
    mask_i = DataForStat2.bwMaskLowFinal{i};
    img_i = zeros(size(mask_i)); % initialize background to 0
    labels = yBinary(sliceStarts(i):sliceEnds(i));
    rows   = DataForStat2.row_pixCurFinal{i};
    cols   = DataForStat2.col_pixCurFinal{i};
    for j = 1:numel(labels)
        img_i(rows(j), cols(j)) = labels(j);
    end
    result{i} = img_i;
end

% 5. Display the binary classification results side by side
indices = 1:nSlices; % 1:14

figure('Units','inches','Position',[1 1 16 5]);
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
    if DataForStat2.cases{i} == 'CN'
        title({'CN', pixText }, 'FontName', 'Arial');
    else
        title({'AD', pixText }, 'FontName', 'Arial');
    end
end

% 6. Global title
sgtitle('Binary Classified pTau', 'FontName', 'Arial');

% 7. Export high-resolution figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/external dataset', 'pTau_binary_classified_external.png'), '-dpng', '-r600')

%% Complete regression reconstruction and plotting script

% 1. Load DataForStat2
% load("external dataset/DataForStat_120825.mat");

% 2. Load regression dataset
load('external dataset/regression_data_T1D_external.mat', 'y');

% 3. Compute start and end indices for each slice in y vector
nSlices = numel(DataForStat2.bwMaskLowFinal);
sliceStarts = zeros(nSlices, 1);
sliceEnds   = zeros(nSlices, 1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat2.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

% 4. Reconstruct each slice as a 125×89 image (ROI filled with 0/1, background is 0)
result = cell(1, nSlices);
for i = 1:nSlices
    mask_i = DataForStat2.bwMaskLowFinal{i};
    img_i = zeros(size(mask_i)); % initialize background to 0
    labels = y(sliceStarts(i):sliceEnds(i));
    rows   = DataForStat2.row_pixCurFinal{i};
    cols   = DataForStat2.col_pixCurFinal{i};
    for j = 1:numel(labels)
        img_i(rows(j), cols(j)) = labels(j);
    end
    result{i} = img_i;
end

% 5. Display the binary classification results side by side
indices = 1:nSlices; % [1 2 3 4 5 6]

figure('Units','inches','Position',[1 1 16 5]);
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
    if DataForStat2.cases{i} == 'CN'
        title({'CN'}, 'FontName', 'Arial');
    else
        title({'AD'}, 'FontName', 'Arial');
    end
end

% 6. Global title
sgtitle('Continuous pTau', 'FontName', 'Arial');

% 7. Export high-resolution figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/external dataset', 'pTau_continuous_external.png'), '-dpng', '-r600')

%% Three‐Class Dataset Creation with Class‐Mean Target Values (Using Otsu Thresholding)

% List of the four datasets to process
fileList = {'external dataset/regression_data_T1D_external.mat', 'external dataset/regression_image_data_T1D_external.mat'};

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
exportgraphics(gcf, fullfile('image/dataset', 'histogram_y_otsu_ternary_external.png'), 'Resolution', 300);

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
    if endsWith(newName, '_external')
        newName = extractBefore(newName, strlength(newName) - strlength('_external') + 1);
    end
    outFile = fullfile(pathstr, [newName '_ternary_external.mat']);

    % 4f. Save the feature matrix and the new continuous target
    save(outFile, 'X', 'yClassMean', 'thresholds', '-v7.3');
    fprintf('Saved class‐mean target dataset to "%s"\n', outFile);
end

%% Visualization of ternary y

% Load the data of T1D as an example
load('external dataset/classification_data_T1D_ternary_external.mat');

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
print(gcf, fullfile('image/external dataset', 'y_distribution_ternary_external.png'), '-dpng', '-r600')

%% Complete ternary classification reconstruction and plotting script

% 1. Load DataForStat2
% load("external dataset/DataForStat_120825.mat");

% 2. Load binary classification dataset
%   X        [N×2500]  % (unused)
%   yClassMean  [N×1]  % average value of each class
load('external dataset/classification_data_T1D_ternary_external.mat');

% 3. Compute start and end indices for each slice in yClassMean vector
nSlices = numel(DataForStat2.bwMaskLowFinal);
sliceStarts = zeros(nSlices, 1);
sliceEnds   = zeros(nSlices, 1);
offset = 0;
for i = 1:nSlices
    nv = numel(DataForStat2.row_pixCurFinal{i});
    sliceStarts(i) = offset + 1;
    sliceEnds(i)   = offset + nv;
    offset = offset + nv;
end

% 4. Reconstruct each slice as a 125×89 binary image (ROI filled with 0/1, background is 0)
result = cell(1, nSlices);
for i = 1:nSlices
    mask_i = DataForStat2.bwMaskLowFinal{i};
    img_i = zeros(size(mask_i)); % initialize background to 0
    labels = yClassMean(sliceStarts(i):sliceEnds(i));
    rows   = DataForStat2.row_pixCurFinal{i};
    cols   = DataForStat2.col_pixCurFinal{i};
    for j = 1:numel(labels)
        img_i(rows(j), cols(j)) = labels(j);
    end
    result{i} = img_i;
end

% 5. Display the binary classification results side by side
indices = 1:nSlices; % [1 … 14]

figure('Units','inches','Position',[1 1 16 5]);
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
    if DataForStat2.cases{i} == 'CN'
        title({ 'CN', pixText }, 'FontName', 'Arial');
    else
        title({ 'AD', pixText }, 'FontName', 'Arial');
    end
end

% 6. Global title
sgtitle('Ternary Classified pTau', 'FontName', 'Arial');

% 7. Export high-resolution figure
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile('image/external dataset', 'pTau_ternary_classified_external.png'), '-dpng', '-r600')


