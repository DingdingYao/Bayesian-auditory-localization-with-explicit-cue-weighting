%% Figure 3: ECW versus Barumerli 2023 model comparison
% This script generates the 2x2 target-response scatter plot used in Fig. 3,
% comparing the proposed ECW model with the Barumerli 2023 benchmark for a
% representative listener.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Auditory Modeling Toolbox (AMT) and its startup script `amt_start`
%   - Result file `data_Optimization/Best_Lambda_Weights_Strategy.mat`
%   - barumerli2023 calibration/data available via `amt_load`
%
% USAGE:
%   Run this script after the optimization and lambda selection pipeline.
%   It loads the selected ECW parameters and benchmark calibration, then
%   evaluates both models on the same PSG input representation for one
%   representative listener.
%
% OUTPUTS:
%   - Publication-ready target-response comparison plot for Figure 3
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m, Figure1.m, Figure2.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear all; close all; clc;

if ~exist('amt_start', 'file'), error('Audio Modeling Toolbox startup script not found.'); end
amt_start;

subject_id = 8;    % listener ID for the figure
nh_id = 'NH16';    % matching subject identifier
s_idx_plot = 3;    % index in the original 5-subject calibration array of barumerli2023

% ===================== 1. Prepare model data and benchmark configuration =====================
feature_type = 'pge'; % use the PSG (pge) feature representation for both models

fprintf('>>> [1/3] Loading model configuration...\n');

% Load the ECW model fit results for the proposed model
base_dir = 'data_Optimization';
mat_file = fullfile(base_dir, 'Best_Lambda_Weights_Strategy.mat');
if ~exist(mat_file, 'file')
    error('Required optimization result file not found: %s', mat_file);
end
opt_ECW = load(mat_file); 
opt_ECW = opt_ECW.All_Best_Results;

% Load the barumerli2023 benchmark calibration for the PSG representation
calibrations = amt_load('barumerli2023', 'barumerli2023_calibration.mat');
calibrations = calibrations.cache.value;
baru_feat_idx = find(strcmp(calibrations.combination(:,1), feature_type));
if isempty(baru_feat_idx), error('barumerli2023 calibration did not contain pge features.'); end

% ===================== 2. Load behavioral and PSG target data =====================
fprintf('\n>>> Loading behavioral data and PSG target features...\n');

% subject_id and nh_id are set in the configuration section above

% Load the empirical data for the representative listener
data_plot = data_majdak2010('Learn_M');
m_real_plot = data_plot(subject_id).mtx;

% Convert the raw target directions to Cartesian coordinates
raw_targets = m_real_plot(:, 1:2);
obj_target = barumerli2023_coordinates([raw_targets, ones(size(raw_targets,1), 1)], 'spherical');
targets_Cart = obj_target.return_positions('cartesian');

% Extract the PSG stimulus features for this listener's HRTF
sofa_obj = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', data_plot(subject_id).id));
[template_par, ~] = barumerli2023_featureextraction(sofa_obj, feature_type);
[target_ext] = barumerli2023_featureextraction(sofa_obj, feature_type, 'target');

template_feat = [template_par.itd, template_par.ild, template_par.monaural];
TargetAll_feat = [target_ext.itd, target_ext.ild, target_ext.monaural];

% Match each target location to the best PSG target feature vector
Tgt_Cart_All = target_ext.coords.return_positions('cartesian');
CosSim = targets_Cart * Tgt_Cart_All';
[~, best_idx] = max(CosSim, [], 2);
part_feat_raw = TargetAll_feat(best_idx, :);

% Generate ECW model predictions
Tmpl = struct();
Tmpl.coords_cart = template_par.coords.return_positions('cartesian');
Tmpl.grid_el = template_par.coords.return_positions('spherical'); Tmpl.grid_el = Tmpl.grid_el(:,2);
Tmpl.mu = mean(template_feat); Tmpl.std = std(template_feat) + eps;
Tmpl.features = bsxfun(@rdivide, bsxfun(@minus, template_feat, Tmpl.mu), Tmpl.std);

% Note: target features are left in raw units here; the ECW simulation
% function handles normalization consistently with yao2026.m.
Target_ecw = struct('features', part_feat_raw, 'target_cart', targets_Cart);

full_params_ECW = opt_ECW.(sprintf('Sub%d', subject_id)).full_params;
num_alphas = length(full_params_ECW) - 2;
summary_ecw = struct('alphas', full_params_ECW(1:num_alphas), 'prior', full_params_ECW(end-1), 'motion', full_params_ECW(end));

% Simulate the ECW model target-response matrix for the selected listener
m_ecw_plot = yao2026(summary_ecw, Tmpl, Target_ecw);

% Generate barumerli2023 benchmark predictions
num_itd = size(target_ext.itd, 2); num_ild = size(target_ext.ild, 2); num_mon = size(target_ext.monaural, 2);
tgt_baru_struct = struct();
tgt_baru_struct.itd = part_feat_raw(:, 1:num_itd);
tgt_baru_struct.ild = part_feat_raw(:, num_itd+1:num_itd+num_ild);
tgt_baru_struct.monaural = part_feat_raw(:, end-num_mon+1:end);
tgt_baru_struct.coords = obj_target;

% Use the benchmark's calibration sigmas for the selected PSG feature set.
sigma_l = calibrations.sigma(s_idx_plot, baru_feat_idx).values(1);
sigma_l2 = calibrations.sigma(s_idx_plot, baru_feat_idx).values(2);
sigma_mon = calibrations.sigma(s_idx_plot, baru_feat_idx).values(3);
sigma_m = calibrations.sigma(s_idx_plot, baru_feat_idx).values(4);
sigma_prior = calibrations.sigma(s_idx_plot, baru_feat_idx).values(5);
if sigma_mon == 0, sigma_mon = []; end

% Generate the barumerli2023 benchmark predictions for the same targets and PSG features.
m_baru_plot = barumerli2023('template', template_par, 'target', tgt_baru_struct, ...
    'num_exp', 1, ...
    'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, ...
    'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m,...
    'MAP', 'sigma_prior', sigma_prior);

% ===================== 3. Create the 2x2 model comparison scatter plot =====================
fig_name = sprintf('Model Comparison - Subject %s', nh_id);
fig = figure('Name', fig_name, 'Position', [100, 100, 850, 780], 'Color', 'w');

% Layout positions for the four panels (two rows, two columns)
left_col = 0.13; right_col = 0.54; width = 0.40;
top_row = 0.50;  bot_row = 0.08;   height = 0.38;

% --- Panel 1: ECW Lateral (Top Left) ---
ax1 = axes('Position', [left_col, top_row, width, height]);
[h_real, h_sim] = plot_single_panel(ax1, m_real_plot, m_ecw_plot, 'Lateral', 'Proposed Model', 'Lateral');
set(ax1, 'XTickLabel', '');

% --- Panel 2: ECW Polar (Top Right) ---
ax2 = axes('Position', [right_col, top_row, width, height]);
plot_single_panel(ax2, m_real_plot, m_ecw_plot, 'Polar', '', 'Polar');
set(ax2, 'XTickLabel', '', 'YTickLabel', '');

% --- Panel 3: Barumerli Lateral (Bottom Left) ---
ax3 = axes('Position', [left_col, bot_row, width, height]);
plot_single_panel(ax3, m_real_plot, m_baru_plot, 'Lateral', 'barumerli2023', '');
xlabel('Target Angle (deg)', 'FontSize', 13, 'FontWeight', 'bold');

% --- Panel 4: Barumerli Polar (Bottom Right) ---
ax4 = axes('Position', [right_col, bot_row, width, height]);
plot_single_panel(ax4, m_real_plot, m_baru_plot, 'Polar', '', '');
xlabel('Target Angle (deg)', 'FontSize', 13, 'FontWeight', 'bold');
set(ax4, 'YTickLabel', '');

% --- legend for the top-left panel ---
lgd = legend(ax1, [h_real, h_sim], {'Actual human', 'Model prediction'}, ...
    'Position', [0.35, 0.94, 0.3, 0.04], ...
    'Orientation', 'horizontal', 'FontSize', 13, 'Box', 'off');

% ===================== 4. Export Figure 3 to PDF =====================
fprintf('>>> Exporting the figure to publication-ready files (Figure3.pdf)...\n');

% Adjust figure paper size to match the onscreen size, avoiding default A4 scaling
set(fig, 'Units', 'Inches');
pos = get(fig, 'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

% Use painters renderer for high-quality vector output
print(fig, 'Figure3.pdf', '-dpdf', '-painters');

fprintf('>>> Export complete.\n');


%% ===================== helper function 1: plot one target-response panel =====================
function [h_real, h_sim] = plot_single_panel(ax, m_real, m_sim, type, row_title, col_title)
axes(ax); hold on;
Size = 25; font_sz = 11;

if strcmp(type, 'Lateral')
    plot([-100 100], [-100 100], '-', 'Color', [1 1 1]*0.6, 'LineWidth', 1.2);
    
    x_real = m_real(:, 5); y_real = m_real(:, 7);
    x_sim = m_sim(:, 5);   y_sim = m_sim(:, 7);
    
    h_real = scatter(x_real, y_real, Size, 0.6*[1 1 1], 'filled');
    h_sim  = scatter(x_sim, y_sim, Size, 'k');
    
    plot(0, 0, 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    
    axis equal; grid on; box on;
    set(gca, 'XLim', [-100 100], 'YLim', [-100 100], 'XTick', [-90, 0, 90], 'YTick', [-90, 0, 90], 'FontSize', font_sz);
    ylabel('Response Angle (deg)', 'FontSize', 13, 'FontWeight', 'bold');
    
elseif strcmp(type, 'Polar')
    plot([-90 270], [-90 270], '-', 'Color', [1 1 1]*0.6, 'LineWidth', 1.2);
    plot([-90 270], [-90 270]+90, '--', 'Color', [0 0 1]*0.4, 'LineWidth', 1.2);
    plot([-90 270], [-90 270]-90, '--', 'Color', [0 0 1]*0.4, 'LineWidth', 1.2);
    
    x_real = m_real(:, 6); y_real = m_real(:, 8);
    x_sim = m_sim(:, 6);   y_sim = m_sim(:, 8);
    
    valid_pol = (x_sim > -30) & (x_sim < 210);
    x_sim_filt = x_sim(valid_pol); y_sim_filt = y_sim(valid_pol);
    
    h_real = scatter(x_real, y_real, Size, 0.6*[1 1 1], 'filled');
    h_sim  = scatter(x_sim_filt, y_sim_filt, Size, 'k');
    
    plot(0, 0, 'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    
    axis equal; grid on; box on;
    set(gca, 'XLim', [-90 270], 'YLim', [-90 270], 'XTick', [-90, 0, 90, 180, 270], 'YTick', [-90, 0, 90, 180, 270], 'FontSize', font_sz);
end

if ~isempty(row_title)
    text(-0.35, 0.5, row_title, 'Units', 'normalized', 'Rotation', 90, ...
        'HorizontalAlignment', 'center', 'FontSize', 15, 'FontWeight', 'bold');
end

if ~isempty(col_title)
    title(col_title, 'FontSize', 15, 'FontWeight', 'bold');
end
end

