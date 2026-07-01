%% Table1: SOTA benchmark of ECW versus Barumerli 2023
% This script evaluates model predictions for the proposed ECW
% model and the Barumerli 2023 benchmark using exact trial-level PSG
% feature matching. It computes the behavioral metrics reported in the
% manuscript and prints a LaTeX-ready summary for Table 1.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Auditory Modeling Toolbox (AMT) and its startup script `amt_start`
%   - Result file `data_Optimization/Best_Lambda_Weights_Strategy.mat`
%   - Barumerli 2023 calibration data available via `amt_load`
%
% USAGE:
%   Run this script after completing the optimization and lambda selection
%   pipeline. It loads the selected ECW parameters and computes the model
%   performance metrics for direct comparison with the benchmark.
%
% OUTPUTS:
%   - Console summary of Table 1 results
%   - LaTeX-ready text printed for manuscript inclusion
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m, Figure1.m, Figure2.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear all; close all; clc;

if ~exist('amt_start', 'file'), error('Please install the AMT toolbox or add it to the MATLAB path'); end
amt_start;

% ===================== 1. Experimental setup and model initialization =====================
subject_ids = [6, 7, 8, 9, 10];
num_sbj = length(subject_ids);
total_runs = 100;     % repeated runs for prediction averaging
feature_type = 'pge'; % use PSG (positive spectral gradient) features for both models
num_metrics = 6;      % corresponds to [LE, PE, QE, accL, accP, gainP]

fprintf('>>> [1/3] Loading model parameters and feature calibration...\n');

% Load the proposed ECW model parameters
base_dir = 'data_Optimization';        
mat_file = fullfile(base_dir, 'Best_Lambda_Weights_Strategy.mat');
if ~exist(mat_file, 'file')
    error('Required optimization result file not found: %s', mat_file);
end
opt_ECW = load(mat_file); 
opt_ECW = opt_ECW.All_Best_Results;

% Load the Barumerli 2023 PSG benchmark calibration
calibrations = amt_load('barumerli2023', 'barumerli2023_calibration.mat');
calibrations = calibrations.cache.value;
baru_feat_idx = find(strcmp(calibrations.combination(:,1), feature_type));
if isempty(baru_feat_idx), error('Barumerli calibration could not find the pge feature index'); end

% ===================== 2. evaluation loop =====================
res_ECW  = zeros(num_sbj, num_metrics); 
res_BARU = zeros(num_sbj, num_metrics);
res_HUM  = zeros(num_sbj, num_metrics);

% Load Majdak data once before the loop to avoid redundant memory allocation
full_data_table = data_majdak2010('Learn_M');

fprintf('>>> [2/3] Starting evaluation loop for %d repeated runs...\n\n', total_runs);

for s_idx = 1:num_sbj
    curr_id = subject_ids(s_idx);
    fprintf('Processing subject -> ID %d (%d/%d)...\n', curr_id, s_idx, num_sbj);
    
    % Extract this subject's data from the pre-loaded table
    data_majdak = full_data_table(curr_id);
    
    % Use the full raw target set
    targets_deg = data_majdak.mtx(:, 1:2);
    responses_deg = data_majdak.mtx(:, 3:4);
    
    obj_target = barumerli2023_coordinates([targets_deg, ones(size(targets_deg,1), 1)], 'spherical');
    obj_resp = barumerli2023_coordinates([responses_deg, ones(size(responses_deg,1), 1)], 'spherical');
    Data.targets_Cart = obj_target.return_positions('cartesian');
    
    t_sph = obj_target.return_positions('spherical'); r_sph = obj_resp.return_positions('spherical');
    t_lat = obj_target.return_positions('horizontal-polar'); r_lat = obj_resp.return_positions('horizontal-polar');
    m_full = [t_sph(:,1:2), r_sph(:,1:2), t_lat(:,1:2), r_lat(:,1:2)];
    
    % Compute and store the behavioral metrics for this subject (order: LE, PE, QE, accL, accP, gainP)
    beh = barumerli2023_metrics(m_full, 'middle_metrics');
    res_HUM(s_idx, :) = [beh.rmsL, beh.rmsP, beh.querr, beh.accL, beh.accP, beh.gainP];
    
    sofa_obj = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', data_majdak.id));
    
    % Use consistent raw target matching across both model predictions
    % 1500 candidate target locations
    [template_par, ~] = barumerli2023_featureextraction(sofa_obj, feature_type);
    
    % extract raw 1550-target PSG feature set for exact trial matching
    [target_ext] = barumerli2023_featureextraction(sofa_obj, feature_type, 'target');
    
    template_feat = [template_par.itd, template_par.ild, template_par.monaural];
    TargetAll_feat = [target_ext.itd, target_ext.ild, target_ext.monaural]; 
    
    num_itd = size(target_ext.itd, 2); 
    num_ild = size(target_ext.ild, 2); 
    num_mon = size(target_ext.monaural, 2);
    
    % Find the exact matching raw target position for each experimental trial
    Tgt_Cart_All = target_ext.coords.return_positions('cartesian');
    CosSim = Data.targets_Cart * Tgt_Cart_All';
    [~, best_idx] = max(CosSim, [], 2);
    
    % Collect the raw target features that correspond to each experimental trial
    part_feat_raw = TargetAll_feat(best_idx, :);

    % Match ECW and Barumerli model inputs for the same target representation
    % build ECW template structure
    Tmpl = struct(); 
    Tmpl.coords_cart = template_par.coords.return_positions('cartesian');
    Tmpl.grid_el = template_par.coords.return_positions('spherical'); Tmpl.grid_el = Tmpl.grid_el(:,2);
    Tmpl.mu = mean(template_feat); 
    Tmpl.std = std(template_feat) + eps;
    Tmpl.features = bsxfun(@rdivide, bsxfun(@minus, template_feat, Tmpl.mu), Tmpl.std);
    
    % Keep ECW target features in raw units, matching the original yao2026.m internal representation
    Target_ecw = struct('features', part_feat_raw, 'target_cart', Data.targets_Cart);
    
    full_params_ECW = opt_ECW.(sprintf('Sub%d', curr_id)).full_params;
    num_alphas = length(full_params_ECW) - 2;
    summary_ecw = struct('alphas', full_params_ECW(1:num_alphas), 'prior', full_params_ECW(end-1), 'motion', full_params_ECW(end));
    
    % Construct the Barumerli target structure
    tgt_baru_struct = struct();
    tgt_baru_struct.itd = part_feat_raw(:, 1:num_itd);
    tgt_baru_struct.ild = part_feat_raw(:, num_itd+1:num_itd+num_ild);
    tgt_baru_struct.monaural = part_feat_raw(:, end-num_mon+1:end);
    tgt_baru_struct.coords = obj_target; % use the exact same target coordinates as the human experiment
    
    sigma_l = calibrations.sigma(s_idx, baru_feat_idx).values(1);
    sigma_l2 = calibrations.sigma(s_idx, baru_feat_idx).values(2);
    sigma_mon = calibrations.sigma(s_idx, baru_feat_idx).values(3);
    sigma_m = calibrations.sigma(s_idx, baru_feat_idx).values(4);
    sigma_prior = calibrations.sigma(s_idx, baru_feat_idx).values(5);
    if sigma_mon == 0, sigma_mon = []; end

    % Subject-level prediction loop
    val_ecw_l=NaN(total_runs,1); val_ecw_p=NaN(total_runs,1); val_ecw_q=NaN(total_runs,1);
    val_ecw_al=NaN(total_runs,1); val_ecw_ap=NaN(total_runs,1); val_ecw_g=NaN(total_runs,1);
    
    val_baru_l=NaN(total_runs,1); val_baru_p=NaN(total_runs,1); val_baru_q=NaN(total_runs,1);
    val_baru_al=NaN(total_runs,1); val_baru_ap=NaN(total_runs,1); val_baru_g=NaN(total_runs,1);
    
    for run = 1:total_runs
        % ---- ECW run prediction ----
        m_ecw_run = yao2026(summary_ecw, Tmpl, Target_ecw);
        res_e = barumerli2023_metrics(m_ecw_run, 'middle_metrics');
        val_ecw_l(run)=res_e.rmsL; val_ecw_p(run)=res_e.rmsP; val_ecw_q(run)=res_e.querr;
        val_ecw_al(run)=res_e.accL; val_ecw_ap(run)=res_e.accP; val_ecw_g(run)=res_e.gainP;
        
        % ---- Barumerli run prediction ----
        m_baru_run = barumerli2023('template', template_par, 'target', tgt_baru_struct, ...
                                      'num_exp', 1, ... 
                                      'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, ...
                                      'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m, ...
                                      'MAP', 'sigma_prior', sigma_prior);
        res_b = barumerli2023_metrics(m_baru_run, 'middle_metrics');
        val_baru_l(run)=res_b.rmsL; val_baru_p(run)=res_b.rmsP; val_baru_q(run)=res_b.querr;
        val_baru_al(run)=res_b.accL; val_baru_ap(run)=res_b.accP; val_baru_g(run)=res_b.gainP;
    end
    
    % --- Store the averaged metric values: LE, PE, QE, accL, accP, gainP ---
    res_ECW(s_idx, :) = [mean(val_ecw_l(~isnan(val_ecw_l))), mean(val_ecw_p(~isnan(val_ecw_p))), mean(val_ecw_q(~isnan(val_ecw_q))), ...
                         mean(val_ecw_al(~isnan(val_ecw_al))), mean(val_ecw_ap(~isnan(val_ecw_ap))), mean(val_ecw_g(~isnan(val_ecw_g)))];
                         
    res_BARU(s_idx, :) = [mean(val_baru_l(~isnan(val_baru_l))), mean(val_baru_p(~isnan(val_baru_p))), mean(val_baru_q(~isnan(val_baru_q))), ...
                          mean(val_baru_al(~isnan(val_baru_al))), mean(val_baru_ap(~isnan(val_baru_ap))), mean(val_baru_g(~isnan(val_baru_g)))];
end

% ===================== 3. Statistics and output =====================
fprintf('>>> [3/3] Computing group-level statistics...\n');

% Compute absolute errors between model predictions and human data (|Model - Human|)
err_ECW  = abs(res_ECW - res_HUM);
err_BARU = abs(res_BARU - res_HUM);

mean_err_ECW = mean(err_ECW, 1); std_err_ECW = std(err_ECW, 0, 1);
mean_err_BARU = mean(err_BARU, 1); std_err_BARU = std(err_BARU, 0, 1);

fprintf('\n===========================================================================================================\n');
fprintf(' Table 1: Absolute Prediction Errors vs. Actual Human (Mean +/- SD, N=%d, Unpooled %d runs)\n', num_sbj, total_runs);
fprintf(' Note: Both models are driven by the exact SAME raw Positive Spectral Gradient (PSG) features.\n');
fprintf('===========================================================================================================\n');
fprintf(' Model       |   |DLE| (deg)   |   |DPE| (deg)   |    |DQE| (%%)    |    |DaccL|      |    |DaccP|      |   |DgainP|      \n');
fprintf('-----------------------------------------------------------------------------------------------------------\n');
% order: LE(1), PE(2), QE(3), accL(4), accP(5), gainP(6)
fprintf(' barumerli2023 | %5.1f +/- %4.1f | %5.1f +/- %4.1f | %5.1f +/- %4.1f | %5.2f +/- %4.2f | %5.2f +/- %4.2f | %5.2f +/- %4.2f \n', ...
    mean_err_BARU(1), std_err_BARU(1), mean_err_BARU(2), std_err_BARU(2), mean_err_BARU(3), std_err_BARU(3), ...
    mean_err_BARU(4), std_err_BARU(4), mean_err_BARU(5), std_err_BARU(5), mean_err_BARU(6), std_err_BARU(6));
fprintf(' proposed model | %5.1f +/- %4.1f | %5.1f +/- %4.1f | %5.1f +/- %4.1f | %5.2f +/- %4.2f | %5.2f +/- %4.2f | %5.2f +/- %4.2f \n', ...
    mean_err_ECW(1), std_err_ECW(1), mean_err_ECW(2), std_err_ECW(2), mean_err_ECW(3), std_err_ECW(3), ...
    mean_err_ECW(4), std_err_ECW(4), mean_err_ECW(5), std_err_ECW(5), mean_err_ECW(6), std_err_ECW(6));
fprintf('===========================================================================================================\n\n');
