%% Figure 4: Own-vs-Other HRTF cross-validation for localization performance
% This script computes Own-vs-Other localization metrics for the proposed ECW
% model and benchmark references, then generates the manuscript-ready Figure 4
% panels showing own/other performance across localization metrics.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Auditory Modeling Toolbox (AMT) and its startup script `amt_start`
%   - Result file `data_Optimization/Best_Lambda_Weights_Strategy.mat`
%   - barumerli2023 calibration/data available via `amt_load`
%
% USAGE:
%   Run this script after the optimization and lambda selection pipeline.
%   It loads the selected ECW parameter sets, runs Own-vs-Other cross-
%   validation simulations, and plots the resulting localization metrics.
%
% OUTPUTS:
%   - Figure4.pdf : manuscript-ready figure file
%   - Figure4.mat : cached Own-vs-Other simulation results
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m, Figure1.m, Figure2.m, Figure3.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear all; close all; clc;

if ~exist('amt_start', 'file'), error('Please install the AMT toolbox or add it to the MATLAB path'); end
amt_start;

% ===================== 1. Experimental settings =====================
feature_type = 'pge';
subject_ids = [6, 7, 8, 9, 10]; 
sbj_num = length(subject_ids);
total_runs = 50; % number of repeated simulation runs
quants = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1];

% ===================== 2. Load and prepare data =====================
cache_filename = 'Figure4.mat';

if exist(cache_filename, 'file')
    fprintf('\n>>> [Cache] Found precomputed data file %s\n', cache_filename);
    fprintf('>>> Loading cached simulation results to skip recomputation...\n');
    load(cache_filename);
else
    fprintf('\n>>> [Cache miss] %s not found, starting full 5x5 Own-vs-Other simulation...\n', cache_filename);
    
    % --- load optimized ECW parameters and Barumerli calibration ---
    fprintf('  -> Loading the ECW optimization results and Barumerli calibration data...\n');
    base_dir = 'data_Optimization';
    mat_file = fullfile(base_dir, 'Best_Lambda_Weights_Strategy.mat');
    if ~exist(mat_file, 'file')
        error('Required optimization result file not found: %s', mat_file);
    end
    load(mat_file, 'All_Best_Results');

    calibrations = amt_load('barumerli2023', 'barumerli2023_calibration.mat');
    calibrations = calibrations.cache.value;
    baru_feat_idx = find(strcmp(calibrations.combination(:,1), feature_type));
    
    % --- load HRTF data for the 1500-location grid and prepare templates ---
    fprintf('  -> Loading the HRTF template set and preparing model inputs...\n');
    Models_ECW = cell(sbj_num, 1); Templates_ECW = cell(sbj_num, 1); Targets_ECW = cell(sbj_num, 1);
    Templates_Baru = cell(sbj_num, 1); Targets_Baru = cell(sbj_num, 1); Baru_Sigmas = cell(sbj_num, 1);
    full_data_table = data_majdak2010('Learn_M');

    for s = 1:sbj_num
        curr_id = subject_ids(s); sbj_id_str = full_data_table(curr_id).id;
        
        % Retrieve the listener-specific ECW parameters from optimization
        global_full_params = All_Best_Results.(sprintf('Sub%d', curr_id)).full_params;
        Models_ECW{s}.alphas = global_full_params(1:29);
        Models_ECW{s}.prior = global_full_params(30); Models_ECW{s}.motion = global_full_params(31);
        
        % Retrieve listener-specific Barumerli calibration sigmas for the chosen feature type
        s_idx_calib = find(strcmpi(calibrations.name, sbj_id_str));
        Baru_Sigmas{s} = calibrations.sigma(s_idx_calib, baru_feat_idx).values;
        
        % Load the listener's individualized SOFA HRTF and extract features
        sofa_obj = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', sbj_id_str));
        [template_par, target_ext] = barumerli2023_featureextraction(sofa_obj, feature_type);
        
        Templates_Baru{s} = template_par; Targets_Baru{s} = target_ext;
        template_feat = [template_par.itd, template_par.ild, template_par.monaural];
        
        % Store raw ECW target features and Cartesian target coordinates
        Targets_ECW{s}.features = [target_ext.itd, target_ext.ild, target_ext.monaural];
        Targets_ECW{s}.target_cart = target_ext.coords.return_positions('cartesian');
        
        % Construct the ECW template representation with internal Z-score normalization
        Tmpl = struct(); Tmpl.coords_cart = template_par.coords.return_positions('cartesian');
        Tmpl.grid_el = template_par.coords.return_positions('spherical'); Tmpl.grid_el = Tmpl.grid_el(:, 2);
        Tmpl.mu = mean(template_feat); Tmpl.std = std(template_feat) + eps;
        Tmpl.features = bsxfun(@rdivide, bsxfun(@minus, template_feat, Tmpl.mu), Tmpl.std);
        Templates_ECW{s} = Tmpl;
    end

    % --- run real 5x5 Own-vs-Other evaluation with repeated simulation ---
    fprintf('  -> Starting 5x5 Own-vs-Other cross-validation (simulation repeats: %d)...\n', total_runs);
    metrics_ecw = repmat(struct('rmsL', 0, 'rmsP', 0, 'querr', 0, 'accL', 0, 'accP', 0, 'gainP', 0), sbj_num, sbj_num);
    metrics_baru = repmat(struct('rmsL', 0, 'rmsP', 0, 'querr', 0, 'accL', 0, 'accP', 0, 'gainP', 0), sbj_num, sbj_num);

    for s = 1:sbj_num 
        for j = 1:sbj_num 
            fprintf('     Processing listener %d with sound set %d ... ', subject_ids(s), subject_ids(j));
            
            % Initialize arrays for the repeated simulation runs
            val_ecw_l=NaN(total_runs,1); val_ecw_p=NaN(total_runs,1); val_ecw_q=NaN(total_runs,1);
            val_ecw_al=NaN(total_runs,1); val_ecw_ap=NaN(total_runs,1); val_ecw_g=NaN(total_runs,1);
            
            val_baru_l=NaN(total_runs,1); val_baru_p=NaN(total_runs,1); val_baru_q=NaN(total_runs,1);
            val_baru_al=NaN(total_runs,1); val_baru_ap=NaN(total_runs,1); val_baru_g=NaN(total_runs,1);
            
            sigma_l = Baru_Sigmas{s}(1); sigma_l2 = Baru_Sigmas{s}(2);
            sigma_mon = Baru_Sigmas{s}(3); sigma_m = Baru_Sigmas{s}(4); sigma_prior = Baru_Sigmas{s}(5);
            if sigma_mon == 0, sigma_mon = []; end
            
            % --- start each simulation repeat ---
            for run = 1:total_runs
                % ECW model prediction using yao2026 with listener-specific parameters and template normalization
                m_ecw_run = yao2026(Models_ECW{s}, Templates_ECW{s}, Targets_ECW{j});
                res_ecw = barumerli2023_metrics(m_ecw_run, 'middle_metrics');
                val_ecw_l(run)=res_ecw.rmsL; val_ecw_p(run)=res_ecw.rmsP; val_ecw_q(run)=res_ecw.querr;
                val_ecw_al(run)=res_ecw.accL; val_ecw_ap(run)=res_ecw.accP; val_ecw_g(run)=res_ecw.gainP;
                
                % Barumerli benchmark prediction using the listener-specific calibration sigmas
                m_baru_run = barumerli2023('template', Templates_Baru{s}, 'target', Targets_Baru{j}, ...
                                       'num_exp', 1, 'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, ...
                                       'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m, 'MAP', 'sigma_prior', sigma_prior);
                res_baru = barumerli2023_metrics(m_baru_run, 'middle_metrics');
                val_baru_l(run)=res_baru.rmsL; val_baru_p(run)=res_baru.rmsP; val_baru_q(run)=res_baru.querr;
                val_baru_al(run)=res_baru.accL; val_baru_ap(run)=res_baru.accP; val_baru_g(run)=res_baru.gainP;
            end
            
            % --- average the repeated simulation results over the 50 runs ---
            metrics_ecw(s,j).rmsL = mean(val_ecw_l(~isnan(val_ecw_l))); 
            metrics_ecw(s,j).rmsP = mean(val_ecw_p(~isnan(val_ecw_p))); 
            metrics_ecw(s,j).querr = mean(val_ecw_q(~isnan(val_ecw_q)));
            metrics_ecw(s,j).accL = mean(val_ecw_al(~isnan(val_ecw_al))); 
            metrics_ecw(s,j).accP = mean(val_ecw_ap(~isnan(val_ecw_ap))); 
            metrics_ecw(s,j).gainP = mean(val_ecw_g(~isnan(val_ecw_g)));
            
            metrics_baru(s,j).rmsL = mean(val_baru_l(~isnan(val_baru_l))); 
            metrics_baru(s,j).rmsP = mean(val_baru_p(~isnan(val_baru_p))); 
            metrics_baru(s,j).querr = mean(val_baru_q(~isnan(val_baru_q)));
            metrics_baru(s,j).accL = mean(val_baru_al(~isnan(val_baru_al))); 
            metrics_baru(s,j).accP = mean(val_baru_ap(~isnan(val_baru_ap))); 
            metrics_baru(s,j).gainP = mean(val_baru_g(~isnan(val_baru_g)));
            
            fprintf('Done. (ECW QE: %.1f%%)\n', metrics_ecw(s,j).querr);
        end
    end

    % --- load the published human reference data for the Own/Other paradigm ---
    fprintf('  -> Loading published behavioral reference datasets for comparison...\n');
    data_middle = data_middlebrooks1999;
    data_baum_temp = exp_baumgartner2014('fig9', 'no_plot');
    data_reij_temp = exp_reijniers2014('fig2_barumerli2020forum', 'no_plot');

    % --- save the processed metrics to cache ---
    fprintf('>>> Saving processed metrics to %s\n', cache_filename);
    save(cache_filename, 'metrics_ecw', 'metrics_baru', 'data_middle', 'data_baum_temp', 'data_reij_temp');
end


% ===================== 3. Organize data and compute summary statistics =====================
fprintf('\n>>> Preparing figure data...\n');
own_idx = logical(eye(sbj_num)); other_idx = not(own_idx);

% ECW model statistics
my_le_own = get_quantiles([metrics_ecw(own_idx).rmsL], quants);
my_le_other = get_quantiles([metrics_ecw(other_idx).rmsL], quants);
my_pe_own = get_quantiles([metrics_ecw(own_idx).rmsP], quants);
my_pe_other = get_quantiles([metrics_ecw(other_idx).rmsP], quants);
my_qe_own = get_quantiles([metrics_ecw(own_idx).querr], quants);
my_qe_other = get_quantiles([metrics_ecw(other_idx).querr], quants);

% Barumerli benchmark statistics
baru_le_own = get_quantiles([metrics_baru(own_idx).rmsL], quants);
baru_le_other = get_quantiles([metrics_baru(other_idx).rmsL], quants);
baru_pe_own = get_quantiles([metrics_baru(own_idx).rmsP], quants);
baru_pe_other = get_quantiles([metrics_baru(other_idx).rmsP], quants);
baru_qe_own = get_quantiles([metrics_baru(own_idx).querr], quants);
baru_qe_other = get_quantiles([metrics_baru(other_idx).querr], quants);

% Baumgartner reference statistics
ns = size(data_baum_temp(1).pe, 1);
own_b = eye(ns) == 1; other_b = not(own_b);
data_baum.pe_own = get_quantiles(data_baum_temp(1).pe(own_b), quants);
data_baum.pe_other = get_quantiles(data_baum_temp(1).pe(other_b), quants);
data_baum.qe_own = get_quantiles(data_baum_temp(1).qe(own_b), quants);
data_baum.qe_other = get_quantiles(data_baum_temp(1).qe(other_b), quants);

% Reijniers reference statistics
ns = size(data_reij_temp, 1);
own_r = logical(eye(ns)); other_r = not(own_r);
data_reij.le_own = get_quantiles([data_reij_temp(own_r).rmsL], quants);
data_reij.le_other = get_quantiles([data_reij_temp(other_r).rmsL], quants);
data_reij.pe_own = get_quantiles([data_reij_temp(own_r).rmsP], quants);
data_reij.pe_other = get_quantiles([data_reij_temp(other_r).rmsP], quants);
data_reij.qe_own = get_quantiles([data_reij_temp(own_r).querr], quants);
data_reij.qe_other = get_quantiles([data_reij_temp(other_r).querr], quants);


% ===================== 4. Figure 4 rendering =====================
fprintf('\n>>> Rendering the final Figure 4 panels...\n');

% Plot settings: Define marker style and colors for the manuscript figure
Marker = 's-';
Color_Mid = [0.15, 0.15, 0.15];            % black/gray for actual human data
Color_My_SBL = [0.0000, 0.3500, 0.6500];   % blue for the proposed ECW model
Color_Baru_PGE = [0.8500, 0.2000, 0.1500]; % red for the barumerli2023 benchmark
Color_Baum = [0.4660, 0.6740, 0.1880];     % green for Baumgartner reference
Color_Reij = [0.4940, 0.1840, 0.5560];     % purple for Reijniers reference

dx = 0.13; 
off_mid = -2; off_my = -1; off_b_pge = 0; off_baum = 1; off_reij = 2;

fig = figure('Name', 'Own vs Other Cross-Validation', 'Position', [100, 100, 1200, 320], 'Color', 'w');% 350 -> 320

% Layout settings: Build a three-panel figure for LE, PE, and QE with consistent spacing
left_pad = 0.05; right_pad = 0.02; bot_pad = 0.08; top_pad = 0.02; gap = 0.07;
w_ax = (1 - left_pad - right_pad - 2*gap) / 3;
h_ax = 1 - bot_pad - top_pad;

% --- Panel 1: LE (Lateral Error) ---
ax1 = axes('Position', [left_pad, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax1, 1+off_mid*dx, data_middle.le_own, 'ko-', 8, Color_Mid, Color_Mid);
local_middlebroxplot(ax1, 2+off_mid*dx, data_middle.le_other, 'ko-', 8, Color_Mid, Color_Mid);
local_middlebroxplot(ax1, 1+off_my*dx, my_le_own, Marker, 8, Color_My_SBL, 'w');
local_middlebroxplot(ax1, 2+off_my*dx, my_le_other, Marker, 8, Color_My_SBL, 'w');
local_middlebroxplot(ax1, 1+off_b_pge*dx, baru_le_own, Marker, 8, Color_Baru_PGE, 'w');
local_middlebroxplot(ax1, 2+off_b_pge*dx, baru_le_other, Marker, 8, Color_Baru_PGE, 'w');
local_middlebroxplot(ax1, 1+off_reij*dx, data_reij.le_own, 'v-', 8, Color_Reij, 'w');
local_middlebroxplot(ax1, 2+off_reij*dx, data_reij.le_other, 'v-', 8, Color_Reij, 'w');

set(ax1, 'YLim', [0 45], 'YTick', 0:10:45, 'XTick', 1:2, 'XTickLabel', {'Own', 'Other'}, 'FontSize', 12);
ylabel('Lateral error (deg)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; box on;

% --- Panel 2: PE (Polar Error) ---
ax2 = axes('Position', [left_pad + w_ax + gap, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax2, 1+off_mid*dx, data_middle.pe_own, 'ko-', 8, Color_Mid, Color_Mid);
local_middlebroxplot(ax2, 2+off_mid*dx, data_middle.pe_other, 'ko-', 8, Color_Mid, Color_Mid);
local_middlebroxplot(ax2, 1+off_my*dx, my_pe_own, Marker, 8, Color_My_SBL, 'w');
local_middlebroxplot(ax2, 2+off_my*dx, my_pe_other, Marker, 8, Color_My_SBL, 'w');
local_middlebroxplot(ax2, 1+off_b_pge*dx, baru_pe_own, Marker, 8, Color_Baru_PGE, 'w');
local_middlebroxplot(ax2, 2+off_b_pge*dx, baru_pe_other, Marker, 8, Color_Baru_PGE, 'w');
local_middlebroxplot(ax2, 1+off_baum*dx, data_baum.pe_own, 'd-', 8, Color_Baum, 'w');
local_middlebroxplot(ax2, 2+off_baum*dx, data_baum.pe_other, 'd-', 8, Color_Baum, 'w');
local_middlebroxplot(ax2, 1+off_reij*dx, data_reij.pe_own, 'v-', 8, Color_Reij, 'w');
local_middlebroxplot(ax2, 2+off_reij*dx, data_reij.pe_other, 'v-', 8, Color_Reij, 'w');

set(ax2, 'YLim', [0 65], 'XTick', 1:2, 'XTickLabel', {'Own', 'Other'}, 'FontSize', 12);
ylabel('Polar error (deg)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; box on;

% --- Panel 3: QE (Quadrant Error) ---
ax3 = axes('Position', [left_pad + 2*w_ax + 2*gap, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax3, 1+off_mid*dx, data_middle.qe_own, 'ko-', 8, Color_Mid, Color_Mid);
local_middlebroxplot(ax3, 2+off_mid*dx, data_middle.qe_other, 'ko-', 8, Color_Mid, Color_Mid);
local_middlebroxplot(ax3, 1+off_my*dx, my_qe_own, Marker, 8, Color_My_SBL, 'w');
local_middlebroxplot(ax3, 2+off_my*dx, my_qe_other, Marker, 8, Color_My_SBL, 'w');
local_middlebroxplot(ax3, 1+off_b_pge*dx, baru_qe_own, Marker, 8, Color_Baru_PGE, 'w');
local_middlebroxplot(ax3, 2+off_b_pge*dx, baru_qe_other, Marker, 8, Color_Baru_PGE, 'w');
local_middlebroxplot(ax3, 1+off_baum*dx, data_baum.qe_own, 'd-', 8, Color_Baum, 'w');
local_middlebroxplot(ax3, 2+off_baum*dx, data_baum.qe_other, 'd-', 8, Color_Baum, 'w');
local_middlebroxplot(ax3, 1+off_reij*dx, data_reij.qe_own, 'v-', 8, Color_Reij, 'w');
local_middlebroxplot(ax3, 2+off_reij*dx, data_reij.qe_other, 'v-', 8, Color_Reij, 'w');

set(ax3, 'YLim', [-5 55], 'XTick', 1:2, 'XTickLabel', {'Own', 'Other'}, 'FontSize', 12);
ylabel('Quadrant error (%)', 'FontSize', 14, 'FontWeight', 'bold');
grid on; box on;


% --- build the legend using invisible plot handles for consistent marker/color mapping ---
% Use dummy NaN plots so the legend reflects the panel marker styles without cluttering the actual figure.
axes(ax1); hold on;
h_leg_mid = plot(NaN, NaN, 'ko-', 'LineWidth', 1.25, 'MarkerSize', 6, 'MarkerFaceColor', Color_Mid);
h_leg_my = plot(NaN, NaN, Marker, 'Color', Color_My_SBL, 'LineWidth', 1.25, 'MarkerSize', 6, 'MarkerFaceColor', 'w');
h_leg_baru = plot(NaN, NaN, Marker, 'Color', Color_Baru_PGE, 'LineWidth', 1.25, 'MarkerSize', 6, 'MarkerFaceColor', 'w');
h_leg_baum = plot(NaN, NaN, 'd-', 'Color', Color_Baum, 'LineWidth', 1.25, 'MarkerSize', 6, 'MarkerFaceColor', 'w');
h_leg_reij = plot(NaN, NaN, 'v-', 'Color', Color_Reij, 'LineWidth', 1.25, 'MarkerSize', 6, 'MarkerFaceColor', 'w');

% Use invisible handles to build a clean legend without duplicating plot annotations
lgd = legend([h_leg_mid, h_leg_my, h_leg_baru, h_leg_baum, h_leg_reij], ...
    {'actual human', 'proposed method', 'barumerli2023', 'baumgartner2014', 'reijniers2014'}, ...
    'Location', 'northwest', 'Orientation', 'vertical', 'FontSize', 11);

% Style the legend box for a polished appearance
set(lgd, 'Box', 'on', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1, 'Color', [1 1 1 0.85]);

% ===================== 5. Save the figure and export to PDF/EPS =====================
fprintf('>>> Saving the figure and exporting high-quality output...\n');
set(fig, 'Units', 'Inches');
pos = get(fig, 'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

print(fig, 'Figure4.pdf', '-dpdf', '-painters');
fprintf('>>> Export complete. Open Figure4.pdf in the current folder to inspect the result.\n');

%% ===================== helper functions =====================
function out = get_quantiles(data, quants)
    out.quantiles = quantile(data, quants);
    out.mean = mean(data);
end

function h = local_middlebroxplot(ax, x, data, marker, msize, col_marker, col_fill)
    % 1. Plot the outer quantile whisker line (25th and 75th percentile shading boundaries)
    h = plot(ax, [x x], [data.quantiles(2), data.quantiles(6)], '-', 'Color', col_marker, 'LineWidth', 1.2);
    
    % 2. Plot the inner quantile bar (interquartile range) with thicker line width for emphasis
    plot(ax, [x x], [data.quantiles(3), data.quantiles(5)], '-', 'Color', col_marker, 'LineWidth', 6);
    
    % 3. Plot the mean marker at the center of the quantile summary
    plot(ax, x, data.mean, marker, 'MarkerSize', msize, 'MarkerFaceColor', col_fill, 'MarkerEdgeColor', col_marker, 'LineWidth', 1.5);
end