%% Figure 6: Majdak 2013 spectral degradation experiment (LP and W conditions)
% This script reproduces the spectral degradation experiment from Majdak 2013,
% generating repeated ECW model predictions for broadband, low-pass, and warped
% spatial conditions. It compares the proposed model against Barumerli 2023 and
% Baumgartner 2014 references for participant-specific and pooled predictions.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Auditory Modeling Toolbox (AMT) and its startup script `amt_start`
%   - Result file `data_Optimization/Best_Lambda_Weights_Strategy.mat`
%   - barumerli2023 and baumgartner2014 data available via `amt_load`
%
% USAGE:
%   Run this script after the optimization and lambda selection pipeline.
%   It loads optimized ECW parameters, runs repeated predictions for each
%   spectral degradation condition, and produces the Figure 6 manuscript plot.
%
% OUTPUTS:
%   - Figure6.pdf : manuscript-ready figure file
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m, Figure1.m, Figure2.m, Figure3.m, Figure4.m, Figure5.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear all; close all; clc;

if ~exist('amt_start', 'file'), error('Please install the AMT toolbox or add it to the MATLAB path'); end
amt_start;

% ===================== 1. Simulation setup and cache control =====================
cache_filename = 'Figure6.mat';

if exist(cache_filename, 'file')
        fprintf('\n>>> [Cache] Found cached results %s, loading data and skipping full simulation...\n', cache_filename);
        load(cache_filename);
    else
        fprintf('\n>>> [Cache miss] %s not found, starting full Majdak2013 simulation pipeline...\n', cache_filename);
    feature_type = 'pge';
    subject_ids = [6, 7, 8, 9, 10]; 
    nh_ids = {'NH12', 'NH15', 'NH16', 'NH17', 'NH18'}; 
    num_sbj = length(subject_ids);
    total_runs = 50; % number of repeated prediction runs for averaging
    fs = 48000;
    
    % Baumgartner prediction uses three lateral divisions for PMV evaluation
    latdivision = [-20, 0, 20];
    dlat = 10;
    
    fprintf('>>> [1/4] Loading model parameters and experimental configuration...\n');
    base_dir = 'data_Optimization';
    mat_file = fullfile(base_dir, 'Best_Lambda_Weights_Strategy.mat');
    if ~exist(mat_file, 'file')
        error('Required optimization result file not found: %s', mat_file);
    end
    load(mat_file, 'All_Best_Results');

    calibrations = amt_load('barumerli2023', 'barumerli2023_calibration.mat');
    calibrations = calibrations.cache.value;
    pge_idx = find(strcmp(calibrations.combination(:,1), feature_type));
    if isempty(pge_idx), pge_idx = 2; end

    Conditions = {'BB', 'LP', 'W'};
    N_cond = length(Conditions);
    lp_filter = amt_load('baumgartner2014','spatstrat_lpfilter.mat');

    % ===================== 2. Load behavioral reference data and compute human benchmarks =====================
    fprintf('>>> [2/4] Loading Majdak2013 behavioral data and computing human benchmarks (pool range +-30 deg)...\n');
    
    hum_pe_all = NaN(13, N_cond); hum_qe_all = NaN(13, N_cond); hum_gainp_all = NaN(13, N_cond);
    part_ids = {'NH12', 'NH15'}; 
    part_row_indices = [];

    for C = 1:N_cond
        data_g = data_majdak2013(Conditions{C});
        num_goupell = length(data_g); 
        
        for g = 1:num_goupell
            mtx_curr = data_g(g).mtx;
            
            t_sph = [mtx_curr(:,1:2), ones(size(mtx_curr,1),1)];
            r_sph = [mtx_curr(:,3:4), ones(size(mtx_curr,1),1)];
            obj_t = barumerli2023_coordinates(t_sph, 'spherical');
            obj_r = barumerli2023_coordinates(r_sph, 'spherical');
            t_lat = obj_t.return_positions('horizontal-polar');
            r_lat = obj_r.return_positions('horizontal-polar');
            
            m_hum = [t_sph(:,1:2), r_sph(:,1:2), t_lat(:,1:2), r_lat(:,1:2)];
            res_hum = barumerli2023_metrics(m_hum, 'middle_metrics');
            
            hum_pe_all(g, C) = res_hum.rmsP;
            hum_qe_all(g, C) = res_hum.querr;
            hum_gainp_all(g, C) = res_hum.gainP;
            
            if C == 1 && ismember(data_g(g).id, part_ids)
                part_row_indices(end+1) = g;
            end
        end
    end

    % Initialize model result matrices for pooled and participant-specific conditions
    ecw_pe_pool = NaN(num_sbj, N_cond); ecw_qe_pool = NaN(num_sbj, N_cond); ecw_gainp_pool = NaN(num_sbj, N_cond);
    ecw_pe_part = NaN(num_sbj, N_cond); ecw_qe_part = NaN(num_sbj, N_cond); ecw_gainp_part = NaN(num_sbj, N_cond);
    
    baru_pe_pool = NaN(num_sbj, N_cond); baru_qe_pool = NaN(num_sbj, N_cond); baru_gainp_pool = NaN(num_sbj, N_cond);
    baru_pe_part = NaN(num_sbj, N_cond); baru_qe_part = NaN(num_sbj, N_cond); baru_gainp_part = NaN(num_sbj, N_cond);
    
    baum_pe_pool = NaN(num_sbj, N_cond); baum_qe_pool = NaN(num_sbj, N_cond); baum_gainp_pool = NaN(num_sbj, N_cond);
    baum_pe_part = NaN(num_sbj, N_cond); baum_qe_part = NaN(num_sbj, N_cond); baum_gainp_part = NaN(num_sbj, N_cond);

    % ===================== 3. Run spectral distortion predictions for BB / LP / W =====================
    fprintf('>>> [3/4] Running BB / LP / W model predictions across subjects...\n');
    for i = 1:num_sbj
        curr_id = subject_ids(i); curr_nh = nh_ids{i};
        is_part_subject = (i <= 2); 
        fprintf('\n========== Processing subject %s (SOFA ID: %d) ==========', curr_nh, curr_id);
        
        global_full_params = All_Best_Results.(sprintf('Sub%d', curr_id)).full_params;
        summary.alphas = global_full_params(1:29); summary.prior = global_full_params(30); summary.motion = global_full_params(31);
        
        sigma_l = calibrations.sigma(i, pge_idx).values(1); sigma_l2 = calibrations.sigma(i, pge_idx).values(2);
        sigma_mon = calibrations.sigma(i, pge_idx).values(3); sigma_m = calibrations.sigma(i, pge_idx).values(4); sigma_prior = calibrations.sigma(i, pge_idx).values(5);
        
        data_majdak = data_majdak2010('Learn_M');
        sofa_clean = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', data_majdak(curr_id).id));
        
        % Extract the template feature representation from the clean HRTF (1500 locations)
        [template, ~] = barumerli2023_featureextraction(sofa_clean, feature_type);
        template_feat = [template.itd, template.ild, template.monaural];
        Tmpl = struct(); Tmpl.coords_cart = template.coords.return_positions('cartesian');
        Tmpl.grid_el = template.coords.return_positions('spherical'); Tmpl.grid_el = Tmpl.grid_el(:,2);
        Tmpl.mu = mean(template_feat); Tmpl.std = std(template_feat) + eps;
        Tmpl.features = bsxfun(@rdivide, bsxfun(@minus, template_feat, Tmpl.mu), Tmpl.std);
        
        % Define the pooled median-lateral region for the Majdak2013 pool comparison
        cart_1500 = Tmpl.coords_cart;
        latpol_1500 = template.coords.return_positions('horizontal-polar');
        idx_pool = find(latpol_1500(:, 1) >= -30 & latpol_1500(:, 1) <= 30); 
        obj_pool = barumerli2023_coordinates(cart_1500(idx_pool, :), 'cartesian');
        pool_cart = obj_pool.return_positions('cartesian'); 
        
        % Load subject-specific Baumgartner2014 SP-DTF data for the current listener
        s_obj = data_baumgartner2014('pool'); pool_idx = find(strcmp({s_obj.id}, curr_nh));
        if ~isempty(pool_idx)
            spdtfs_ref = s_obj(pool_idx).Obj; 
            spdtfs_tmpl_cell = cell(1, length(latdivision));
            polang_baum_cell = cell(1, length(latdivision));
            for ii = 1:length(latdivision)
                [spdtfs_tmpl_cell{ii}, polang_baum_cell{ii}] = extractsp(latdivision(ii), spdtfs_ref);
            end
        end
        
        for C = 1:N_cond
            Cond = Conditions{C};
            fprintf('  -> Processing condition %s ... ', Cond);
            sofa_cond = sofa_clean; 
            if ~isempty(pool_idx), spdtfs_cond_cell = spdtfs_tmpl_cell; end
            
            % --- apply spectral distortion conditions to HRTF/SP-DTF data ---
            if strcmp(Cond, 'LP')
                if ~isempty(pool_idx)
                    for ii = 1:length(latdivision)
                        for ang = 1:size(spdtfs_tmpl_cell{ii}, 2)
                            for ch = 1:size(spdtfs_tmpl_cell{ii}, 3)
                                spdtfs_cond_cell{ii}(:, ang, ch) = filter(lp_filter.blp, lp_filter.alp, spdtfs_tmpl_cell{ii}(:, ang, ch));
                            end
                        end
                    end
                end
                for dir = 1:size(sofa_cond.Data.IR,1)
                    sofa_cond.Data.IR(dir, 1, :) = filter(lp_filter.blp, lp_filter.alp, squeeze(sofa_clean.Data.IR(dir, 1, :)));
                    sofa_cond.Data.IR(dir, 2, :) = filter(lp_filter.blp, lp_filter.alp, squeeze(sofa_clean.Data.IR(dir, 2, :)));
                end
                
            elseif strcmp(Cond, 'W')
                if ~isempty(pool_idx)
                    for ii = 1:length(latdivision)
                        spdtfs_cond_cell{ii} = local_warphrtf(spdtfs_tmpl_cell{ii}, fs);
                    end
                end
                hM = permute(sofa_clean.Data.IR, [3, 1, 2]); 
                hM_warped = local_warphrtf(hM, fs);          
                sofa_cond.Data.IR = permute(hM_warped, [2, 3, 1]); 
            end
            
            % --- compute Baumgartner PMV predictions for each lateral division ---
            if ~isempty(pool_idx)
                p_map_cell = cell(1, length(latdivision));
                rang_cell = cell(1, length(latdivision));
                for ii = 1:length(latdivision)
                    [p_map_cell{ii}, rang_cell{ii}] = baumgartner2014(spdtfs_cond_cell{ii}, spdtfs_tmpl_cell{ii}, fs, 'S', s_obj(pool_idx).S, 'lat', latdivision(ii), 'polsamp', polang_baum_cell{ii});
                end
            end
            
            % --- extract target features from the distorted HRTF ---
            [tgt_extraction] = barumerli2023_featureextraction(sofa_cond, feature_type);
            TargetAll_feat = [tgt_extraction.itd, tgt_extraction.ild, tgt_extraction.monaural];
            num_itd = size(tgt_extraction.itd, 2); num_ild = size(tgt_extraction.ild, 2); num_mon = size(tgt_extraction.monaural, 2);
            
            % Pool target features for the aggregated comparison
            pool_feat_raw = TargetAll_feat(idx_pool, :);
            tgt_pool_ecw = struct('features', pool_feat_raw, 'target_cart', pool_cart);
            tgt_pool_baru = struct('itd', pool_feat_raw(:, 1:num_itd), 'ild', pool_feat_raw(:, num_itd+1:num_itd+num_ild), 'monaural', pool_feat_raw(:, end-num_mon+1:end), 'coords', obj_pool);
            
            % Participant-specific target features for the two Majdak listeners
            g_idx = find(strcmp({data_g.id}, curr_nh));
            run_part = is_part_subject && ~isempty(g_idx);
            if run_part
                data_g_temp = data_majdak2013(Cond); 
                g_idx_temp = find(strcmp({data_g_temp.id}, curr_nh));
                mtx_subj = data_g_temp(g_idx_temp).mtx; 
                
                idlat = mtx_subj(:,7) <= 30 & mtx_subj(:,7) >= -30;
                part_az = mtx_subj(idlat, 1); part_el = mtx_subj(idlat, 2); part_pol = mtx_subj(idlat, 6);
                
                % Baumgartner participant-specific target polarization sets
                if ~isempty(pool_idx)
                    part_pol_baum_cell = cell(1, length(latdivision));
                    for ii = 1:length(latdivision)
                        idlat_b = mtx_subj(:,7) <= latdivision(ii)+dlat & mtx_subj(:,7) > latdivision(ii)-dlat;
                        part_pol_baum_cell{ii} = mtx_subj(idlat_b, 6);
                    end
                end
                
                % Find the nearest 1500-grid directions to the original target directions
                obj_part_real = barumerli2023_coordinates([part_az, part_el, ones(length(part_az),1)], 'spherical');
                part_cart_real = obj_part_real.return_positions('cartesian');
                [~, best_idx_part] = max(part_cart_real * cart_1500', [], 2);
                part_feat_raw = TargetAll_feat(best_idx_part, :);
                obj_part_matched = barumerli2023_coordinates(cart_1500(best_idx_part, :), 'cartesian');
                part_cart_matched = obj_part_matched.return_positions('cartesian');
                
                tgt_part_ecw = struct('features', part_feat_raw, 'target_cart', part_cart_matched);
                tgt_part_baru = struct('itd', part_feat_raw(:, 1:num_itd), 'ild', part_feat_raw(:, num_itd+1:num_itd+num_ild), 'monaural', part_feat_raw(:, end-num_mon+1:end), 'coords', obj_part_matched);
            end
            
            % --- [Repeat model predictions for averaging] ---
            val_ecw_pl_pe=NaN(total_runs,1); val_ecw_pl_qe=NaN(total_runs,1); val_ecw_pl_g=NaN(total_runs,1);
            val_baru_pl_pe=NaN(total_runs,1); val_baru_pl_qe=NaN(total_runs,1); val_baru_pl_g=NaN(total_runs,1);
            val_baum_pl_pe=NaN(total_runs,1); val_baum_pl_qe=NaN(total_runs,1); val_baum_pl_g=NaN(total_runs,1);
            
            val_ecw_pa_pe=NaN(total_runs,1); val_ecw_pa_qe=NaN(total_runs,1); val_ecw_pa_g=NaN(total_runs,1);
            val_baru_pa_pe=NaN(total_runs,1); val_baru_pa_qe=NaN(total_runs,1); val_baru_pa_g=NaN(total_runs,1);
            val_baum_pa_pe=NaN(total_runs,1); val_baum_pa_qe=NaN(total_runs,1); val_baum_pa_g=NaN(total_runs,1);
            
            for s = 1:total_runs
                % Pool prediction (aggregate pool condition)
                m_ecw_pool = yao2026(summary, Tmpl, tgt_pool_ecw);
                res = barumerli2023_metrics(m_ecw_pool, 'middle_metrics');
                val_ecw_pl_pe(s)=res.rmsP; val_ecw_pl_qe(s)=res.querr; val_ecw_pl_g(s)=res.gainP;
                
                m_baru_pool = barumerli2023('template', template, 'target', tgt_pool_baru, 'num_exp', 1, 'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, 'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m, 'sigma_prior', sigma_prior);
                res = barumerli2023_metrics(m_baru_pool, 'middle_metrics');
                val_baru_pl_pe(s)=res.rmsP; val_baru_pl_qe(s)=res.querr; val_baru_pl_g(s)=res.gainP;
                
                if ~isempty(pool_idx)
                    m_baum_pool_all = [];
                    for ii = 1:length(latdivision)
                        m_v_p = local_baumgartner2014_virtualexp(p_map_cell{ii}, polang_baum_cell{ii}, rang_cell{ii}, 'lat', latdivision(ii), 'runs', 1);
                        m_baum_pool_all = [m_baum_pool_all; zeros(size(m_v_p, 1), 4), m_v_p(:, 5:8)]; 
                    end
                    res = barumerli2023_metrics(m_baum_pool_all, 'middle_metrics');
                    val_baum_pl_pe(s)=res.rmsP; val_baum_pl_qe(s)=res.querr; val_baum_pl_g(s)=res.gainP;
                end
                
                % Part prediction (participant-specific condition)
                if run_part
                    m_ecw_part = yao2026(summary, Tmpl, tgt_part_ecw);
                    res = barumerli2023_metrics(m_ecw_part, 'middle_metrics');
                    val_ecw_pa_pe(s)=res.rmsP; val_ecw_pa_qe(s)=res.querr; val_ecw_pa_g(s)=res.gainP;
                    
                    m_baru_part = barumerli2023('template', template, 'target', tgt_part_baru, 'num_exp', 1, 'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, 'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m, 'sigma_prior', sigma_prior);
                    res = barumerli2023_metrics(m_baru_part, 'middle_metrics');
                    val_baru_pa_pe(s)=res.rmsP; val_baru_pa_qe(s)=res.querr; val_baru_pa_g(s)=res.gainP;
                    
                    if ~isempty(pool_idx)
                        m_baum_part_all = [];
                        for ii = 1:length(latdivision)
                            if ~isempty(part_pol_baum_cell{ii})
                                m_v_pa = local_baumgartner2014_virtualexp(p_map_cell{ii}, polang_baum_cell{ii}, rang_cell{ii}, 'targetset', part_pol_baum_cell{ii}, 'lat', latdivision(ii), 'runs', 1);
                                m_baum_part_all = [m_baum_part_all; zeros(size(m_v_pa, 1), 4), m_v_pa(:, 5:8)];
                            end
                        end
                        if ~isempty(m_baum_part_all)
                            res = barumerli2023_metrics(m_baum_part_all, 'middle_metrics');
                            val_baum_pa_pe(s)=res.rmsP; val_baum_pa_qe(s)=res.querr; val_baum_pa_g(s)=res.gainP;
                        end
                    end
                end
            end % end 50-run averaging loop
            
            % --- average the 50 repeated measurements ---
            ecw_pe_pool(i,C) = mean(val_ecw_pl_pe(~isnan(val_ecw_pl_pe)));
            ecw_qe_pool(i,C) = mean(val_ecw_pl_qe(~isnan(val_ecw_pl_qe)));
            ecw_gainp_pool(i,C) = mean(val_ecw_pl_g(~isnan(val_ecw_pl_g)));
            
            baru_pe_pool(i,C) = mean(val_baru_pl_pe(~isnan(val_baru_pl_pe)));
            baru_qe_pool(i,C) = mean(val_baru_pl_qe(~isnan(val_baru_pl_qe)));
            baru_gainp_pool(i,C) = mean(val_baru_pl_g(~isnan(val_baru_pl_g)));
            
            if ~isempty(pool_idx)
                baum_pe_pool(i,C) = mean(val_baum_pl_pe(~isnan(val_baum_pl_pe)));
                baum_qe_pool(i,C) = mean(val_baum_pl_qe(~isnan(val_baum_pl_qe)));
                baum_gainp_pool(i,C) = mean(val_baum_pl_g(~isnan(val_baum_pl_g)));
            end
            
            if run_part
                ecw_pe_part(i,C) = mean(val_ecw_pa_pe(~isnan(val_ecw_pa_pe)));
                ecw_qe_part(i,C) = mean(val_ecw_pa_qe(~isnan(val_ecw_pa_qe)));
                ecw_gainp_part(i,C) = mean(val_ecw_pa_g(~isnan(val_ecw_pa_g)));
                
                baru_pe_part(i,C) = mean(val_baru_pa_pe(~isnan(val_baru_pa_pe)));
                baru_qe_part(i,C) = mean(val_baru_pa_qe(~isnan(val_baru_pa_qe)));
                baru_gainp_part(i,C) = mean(val_baru_pa_g(~isnan(val_baru_pa_g)));
                
                if ~isempty(pool_idx)
                    baum_pe_part(i,C) = mean(val_baum_pa_pe(~isnan(val_baum_pa_pe)));
                    baum_qe_part(i,C) = mean(val_baum_pa_qe(~isnan(val_baum_pa_qe)));
                    baum_gainp_part(i,C) = mean(val_baum_pa_g(~isnan(val_baum_pa_g)));
                end
            end
            fprintf('OK\n');
        end
    end

    fprintf('>>> [4/4] Saving computed results to cache...\n');
    save(cache_filename, ...
        'ecw_pe_part', 'ecw_qe_part', 'ecw_gainp_part', 'baru_pe_part', 'baru_qe_part', 'baru_gainp_part', 'baum_pe_part', 'baum_qe_part', 'baum_gainp_part', ...
        'ecw_pe_pool', 'ecw_qe_pool', 'ecw_gainp_pool', 'baru_pe_pool', 'baru_qe_pool', 'baru_gainp_pool', 'baum_pe_pool', 'baum_qe_pool', 'baum_gainp_pool', ...
        'hum_pe_all', 'hum_qe_all', 'hum_gainp_all', 'part_row_indices', 'Conditions', 'N_cond');
    
end

% ===================== 4. Figure 6 rendering (2x3 layout) =====================
fprintf('>>> Rendering the 2x3 figure...\n');

FontSize = 11; MarkerSize = 7; 
% [Plot colors] Keep consistent with the earlier figures
Color_Hum = [0.15, 0.15, 0.15];            
Color_My_ECW = [0.0000, 0.3500, 0.6500];   
Color_Baru = [0.8500, 0.2000, 0.1500];     
Color_Baum = [0.4660, 0.6740, 0.1880];     

% horizontal offsets for overlapping plot markers
dx = 0.12; 
off_hum = -1.5; off_my = -0.5; off_baru = 0.5; off_baum = 1.5;

x_pos = [1, 2, 3]; 
x_ticks = 1:3; x_labels = {'BB', 'LP', 'Warped'};

% Compute robust quantiles for each condition (handles NaN values safely)
hum_pe_pool_q = get_robust_quantiles(hum_pe_all); hum_qe_pool_q = get_robust_quantiles(hum_qe_all); hum_g_pool_q = get_robust_quantiles(hum_gainp_all);
hum_pe_part_q = get_robust_quantiles(hum_pe_all(part_row_indices, :)); hum_qe_part_q = get_robust_quantiles(hum_qe_all(part_row_indices, :)); hum_g_part_q = get_robust_quantiles(hum_gainp_all(part_row_indices, :));

idx_p = [1, 2]; idx_pl = 1:5;
ecw_pe_p_q = get_robust_quantiles(ecw_pe_part(idx_p,:)); baru_pe_p_q = get_robust_quantiles(baru_pe_part(idx_p,:)); baum_pe_p_q = get_robust_quantiles(baum_pe_part(idx_p,:));
ecw_qe_p_q = get_robust_quantiles(ecw_qe_part(idx_p,:)); baru_qe_p_q = get_robust_quantiles(baru_qe_part(idx_p,:)); baum_qe_p_q = get_robust_quantiles(baum_qe_part(idx_p,:));
ecw_g_p_q = get_robust_quantiles(ecw_gainp_part(idx_p,:)); baru_g_p_q = get_robust_quantiles(baru_gainp_part(idx_p,:)); baum_g_p_q = get_robust_quantiles(baum_gainp_part(idx_p,:));

ecw_pe_pl_q = get_robust_quantiles(ecw_pe_pool(idx_pl,:)); baru_pe_pl_q = get_robust_quantiles(baru_pe_pool(idx_pl,:)); baum_pe_pl_q = get_robust_quantiles(baum_pe_pool(idx_pl,:));
ecw_qe_pl_q = get_robust_quantiles(ecw_qe_pool(idx_pl,:)); baru_qe_pl_q = get_robust_quantiles(baru_qe_pool(idx_pl,:)); baum_qe_pl_q = get_robust_quantiles(baum_qe_pool(idx_pl,:));
ecw_g_pl_q = get_robust_quantiles(ecw_gainp_pool(idx_pl,:)); baru_g_pl_q = get_robust_quantiles(baru_gainp_pool(idx_pl,:)); baum_g_pl_q = get_robust_quantiles(baum_gainp_pool(idx_pl,:));

% --- build the 2x3 figure layout ---
fig = figure('Name','Majdak 2013 (BB vs LP vs W)', 'Position',[50 50 1200 520], 'Color','w'); 
left_pad = 0.07; right_pad = 0.02; bot_pad = 0.05; top_pad = 0.05; gap_x = 0.05; gap_y = 0.06;
w_ax = (1 - left_pad - right_pad - 2*gap_x) / 3; h_ax = (1 - bot_pad - top_pad - gap_y) / 2;
y_top = bot_pad + h_ax + gap_y;

% ===================== 4.1 [Row 1: Part] =====================
ax1 = axes('Position', [left_pad, y_top, w_ax, h_ax]); hold on;
local_middlebroxplot(ax1, x_pos+off_baum*dx, baum_pe_p_q, 'pd', MarkerSize, Color_Baum, 'w');
local_middlebroxplot(ax1, x_pos+off_baru*dx, baru_pe_p_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax1, x_pos+off_my*dx, ecw_pe_p_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax1, x_pos+off_hum*dx, hum_pe_part_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
set(ax1, 'XLim', [0.5 3.5], 'XTick', x_ticks, 'XTickLabel', [], 'YLim', [25 62], 'FontSize', FontSize);
ylabel('PE (deg)', 'FontSize', 13, 'FontWeight', 'bold'); 
title('Polar Error (PE)', 'FontWeight', 'bold', 'FontSize', 14); grid on; box on;
text(ax1, -0.18, 0.5, 'Part.', 'Units', 'normalized', 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');

ax2 = axes('Position', [left_pad+w_ax+gap_x, y_top, w_ax, h_ax]); hold on;
local_middlebroxplot(ax2, x_pos+off_baum*dx, baum_qe_p_q, 'pd', MarkerSize, Color_Baum, 'w');
local_middlebroxplot(ax2, x_pos+off_baru*dx, baru_qe_p_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax2, x_pos+off_my*dx, ecw_qe_p_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax2, x_pos+off_hum*dx, hum_qe_part_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
set(ax2, 'XLim', [0.5 3.5], 'XTick', x_ticks, 'XTickLabel', [], 'YLim', [-5 55], 'FontSize', FontSize);
ylabel('QE (%)', 'FontSize', 13, 'FontWeight', 'bold'); 
title('Quadrant Error (QE)', 'FontWeight', 'bold', 'FontSize', 14); grid on; box on;

ax3 = axes('Position', [left_pad+2*w_ax+2*gap_x, y_top, w_ax, h_ax]); hold on;
local_middlebroxplot(ax3, x_pos+off_baum*dx, baum_g_p_q, 'pd', MarkerSize, Color_Baum, 'w');
local_middlebroxplot(ax3, x_pos+off_baru*dx, baru_g_p_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax3, x_pos+off_my*dx, ecw_g_p_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax3, x_pos+off_hum*dx, hum_g_part_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
set(ax3, 'XLim', [0.5 3.5], 'XTick', x_ticks, 'XTickLabel', [], 'YLim', [-0.4 1.2], 'FontSize', FontSize);
ylabel('Gain', 'FontSize', 13, 'FontWeight', 'bold'); 
title('Polar Gain (gainP)', 'FontWeight', 'bold', 'FontSize', 14); grid on; box on;

axes(ax1); hold on;
h_l1 = plot(NaN,NaN,'ko-','LineWidth',1.5,'MarkerSize',MarkerSize+1,'MarkerFaceColor',Color_Hum);
h_l2 = plot(NaN,NaN,'s-','Color',Color_My_ECW,'LineWidth',1.5,'MarkerSize',MarkerSize+1,'MarkerFaceColor','w');
h_l3 = plot(NaN,NaN,'s-','Color',Color_Baru,'LineWidth',1.5,'MarkerSize',MarkerSize,'MarkerFaceColor','w');
h_l4 = plot(NaN,NaN,'d-','Color',Color_Baum,'LineWidth',1.5,'MarkerSize',MarkerSize,'MarkerFaceColor','w');
lgd = legend([h_l1, h_l2, h_l3, h_l4], {'actual human', 'proposed method', 'barumerli2023', 'baumgartner2014'}, ...
    'Location', 'northwest', 'FontSize', 11);
set(lgd, 'Box', 'on', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1, 'Color', [1 1 1 0.85]);

% ===================== 4.2 [Row 2: Pool] =====================
ax4 = axes('Position', [left_pad, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax4, x_pos+off_baum*dx, baum_pe_pl_q, 'pd', MarkerSize, Color_Baum, 'w');
local_middlebroxplot(ax4, x_pos+off_baru*dx, baru_pe_pl_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax4, x_pos+off_my*dx, ecw_pe_pl_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax4, x_pos+off_hum*dx, hum_pe_pool_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
set(ax4, 'XLim', [0.5 3.5], 'XTick', x_ticks, 'XTickLabel', x_labels, 'YLim', [25 62], 'FontSize', FontSize);
ylabel('PE (deg)', 'FontSize', 13, 'FontWeight', 'bold'); grid on; box on;
text(ax4, -0.18, 0.5, 'Pool', 'Units', 'normalized', 'Rotation', 90, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');

ax5 = axes('Position', [left_pad+w_ax+gap_x, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax5, x_pos+off_baum*dx, baum_qe_pl_q, 'pd', MarkerSize, Color_Baum, 'w');
local_middlebroxplot(ax5, x_pos+off_baru*dx, baru_qe_pl_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax5, x_pos+off_my*dx, ecw_qe_pl_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax5, x_pos+off_hum*dx, hum_qe_pool_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
set(ax5, 'XLim', [0.5 3.5], 'XTick', x_ticks, 'XTickLabel', x_labels, 'YLim', [-5 55], 'FontSize', FontSize);
ylabel('QE (%)', 'FontSize', 13, 'FontWeight', 'bold'); grid on; box on;

ax6 = axes('Position', [left_pad+2*w_ax+2*gap_x, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax6, x_pos+off_baum*dx, baum_g_pl_q, 'pd', MarkerSize, Color_Baum, 'w');
local_middlebroxplot(ax6, x_pos+off_baru*dx, baru_g_pl_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax6, x_pos+off_my*dx, ecw_g_pl_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax6, x_pos+off_hum*dx, hum_g_pool_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
set(ax6, 'XLim', [0.5 3.5], 'XTick', x_ticks, 'XTickLabel', x_labels, 'YLim', [-0.4 1.2], 'FontSize', FontSize);
ylabel('Gain', 'FontSize', 13, 'FontWeight', 'bold'); grid on; box on;

% ===================== 5. Save figure output =====================
fprintf('>>> Arranging and printing the figure...\n');
set(fig, 'Units', 'Inches'); pos = get(fig, 'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
print(fig, 'Figure6.pdf', '-dpdf', '-painters');
fprintf('>>> Finished. Open Figure6.pdf to review.\n');

%% ===================== Figure helper definitions =====================

% --- plot helper for middle-brox style plots ---
function local_middlebroxplot(ax, x, data_q, marker, msize, col_marker, col_fill)
    valid_idx = ~isnan(data_q(2,:));
    if any(valid_idx)
        plot(ax, x(valid_idx), data_q(2, valid_idx), '-', 'Color', col_marker, 'LineWidth', 1.5);
    end
    
    m_type = marker(2); 
    for i = 1:length(x)
        q = data_q(:, i);
        if isnan(q(1)), continue; end
        plot(ax, [x(i) x(i)], [q(1), q(3)], '-', 'Color', col_marker, 'LineWidth', 1.5);
        plot(ax, x(i), q(2), m_type, 'MarkerSize', msize, 'MarkerFaceColor', col_fill, 'MarkerEdgeColor', col_marker, 'LineWidth', 1.5);
    end
end

% --- helper: compute robust quantiles ---
function q_matrix = get_robust_quantiles(data_matrix)
    num_conds = size(data_matrix, 2);
    q_matrix = NaN(3, num_conds);
    for c = 1:num_conds
        col = data_matrix(:, c);
        col_clean = col(~isnan(col)); 
        if ~isempty(col_clean)
            q_matrix(:, c) = quantile(col_clean, [.25, .50, .75])';
        end
    end
end

% ===================== helper: Baumgartner virtual experiment wrapper =====================
function m = local_baumgartner2014_virtualexp(p,tang,rang,varargin)
definput.keyvals.runs = 10;
definput.keyvals.targetset = [];
definput.keyvals.lat = 0;
[flags,kv]=ltfatarghelper({'runs','targetset'},definput,varargin);

if isempty(kv.targetset), kv.targetset = tang; end

nt=length(kv.targetset); m = nan(nt*kv.runs,9);
m(:,5) = kv.lat; m(:,6) = repmat(kv.targetset(:),kv.runs,1); m(:,7) = kv.lat;
if length(tang) > 1, tangbound = tang(:)+0.5*diff([tang(1)-diff(tang(1:2));tang(:)]);
else, tangbound = tang; end

post=zeros(nt,1); 
for ii = 1:nt
  if kv.targetset(ii) > max(tangbound), post(ii) = length(tangbound); 
  else, post(ii) = find(tangbound>=kv.targetset(ii),1); end
end

posr=zeros(nt,1);
for rr=1:kv.runs
  for jj = 1:nt 
    posr(jj) = local_discreteinvrnd(p(:,post(jj)),1);
    m(jj+(rr-1)*nt,8) = rang(posr(jj));
  end
end
end

function [ X ] = local_discreteinvrnd(p,n,m)
if ~exist('m','var'), m=1; end
p = p/sum(p); c = cumsum(p); t = max(c)*rand(n,m); 
X = zeros(n,m);
for jj = 1:m
    for ii = 1:n, X(ii,jj) = find(c >= t(ii,jj) ,1); end
end
end

%% ------------------------------------------------------------------------
%  ---- INTERNAL FUNCTIONS ------------------------------------------------
%  ------------------------------------------------------------------------
function hM_warped = local_warphrtf(hM,fs)
% warps HRTFs acc. to Walder (2010)
% Usage: hM_warped = warp_hrtf(hM,fs)

N = fs;
fu = 2800;
fowarped = 8500;
fo = 16000;

fscala = [0:fs/N:fs-fs/N]';
hM_warped = zeros(512,size(hM,2),size(hM,3));
fuindex = max(find(fscala <= fu)); % 2800
fowindex = min(find(fscala >= fowarped));
foindex = min(find(fscala >= fo));

for canal = 1:size(hM,3)
   for el = 1:size(hM,2)
        yi = ones(fs/2+1,1)*(10^-(70/20));
        flin1 = [fscala(1:fuindex-1)];
        flin2 = [linspace(fscala(fuindex),fscala(foindex),fowindex-fuindex+1)]';
        fscalawarped = [flin1 ; flin2];

        % interpolate
        x = fscala(1:foindex);
        H = fft(hM(:,el,canal),N);
        Y = H(1:foindex);
        xi = fscalawarped;
        yi(1:length(xi),1) = interp1(x, Y, xi,'linear');

        yges=([yi; conj(flipud(yi(2:end-1)))]);
        hges = ifft([yges(1:end)],length(yges));
        hges=fftshift(ifft(yges));
        hwin=hges(fs/2-256:fs/2+768);
        hwinfade = local_FWfade(hwin,512,24,96,192);
        hM_warped(1:end,el,canal)=hwinfade;

    end
end
end

function out = local_FWfade(inp, len, fadein, fadeout, offset)
% FW_FADE crop/extend and fade in/out a vector.
%
% OUT = FW_FADE(INP, LEN, FADEIN, FADEOUT, OFFSET) crops or extends with zeros the signal INP
% up to length LEN. Additionally, the result is faded in/out using HANN window with
% the length FADEIN/FADEOUT, respectively. If given, an offset can be added to show
% where the real signal begins.
%
% When used to crop signal, INP is cropped first, then faded out.
% When used to extend signal, INP is faded out first, then extended too provide fading.
%
% INP:     vector with signal (1xN or Nx1)
% LEN:     length of signal OUT (without OFFSET)
% FADEIN:  number of samples to fade in, beginning from OFFSET
% FADEOUT: number of samples to fade out, ending at the end of OUT (without OFFSET)
% OFFSET:  number of offset samples before FADEIN, optional
% OUT:     cropped/extended and faded vector
%
% Setting a parameter to 0 disables corresponding functionality.

% ExpSuite - software framework for applications to perform experiments (related but not limited to psychoacoustics).
% Copyright (C) 2003-2010 Acoustics Research Institute - Austrian Academy of Sciences; Piotr Majdak and Michael Mihocic
% Licensed under the EUPL, Version 1.1 or ? as soon they will be approved by the European Commission - subsequent versions of the EUPL (the "Licence")
% You may not use this work except in compliance with the Licence.
% You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
% Unless required by applicable law or agreed to in writing, software distributed under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the Licence for the specific language governing  permissions and limitations under the Licence.

% 7.11.2003
% 22.08.2005: improvement: INP may be 1xN or Nx1 now.
% Piotr Majdak (piotr@majdak.com)

ss=length(inp);    % get length of inp
	% offset
if ~exist('offset','var')
	offset = 0;
end
offset=round(offset);
len=round(len);
fadein=round(fadein);
fadeout=round(fadeout);
if offset>ss
	error('OFFSET is greater than signal length');
end
	% create new input signal discarding offset
inp2=inp(1+offset:end);
ss=length(inp2);

  % fade in
if fadein ~= 0
  if fadein > ss
    error('FADEIN is greater than signal length');
  end
  han=hanning(fadein*2);
  if size(inp2,1)==1
    han=han';
  end
  inp2(1:fadein) = inp2(1:fadein).*han(1:fadein);
end
  % fade out window

if len == 0
  len = ss;
end
if len <= ss
    % crop and fade out
  out=inp2(1:len);
    % fade out
  if fadeout ~= 0
    if fadeout > len
      error('FADEOUT is greater than cropped signal length');
    end
    han = hanning(2*fadeout);
    if size(out,1)==1
      han=han';
    end
    out(len-fadeout+1:end)=out(len-fadeout+1:end).*han(fadeout+1:end);
  end
else
    % fade out and extend
  if fadeout ~= 0
    if fadeout > ss
      error('FADEOUT is greater than signal length');
    end
    han = hanning(2*fadeout);
    inp2(ss-fadeout+1:end)=inp2(ss-fadeout+1:end).*han(fadeout+1:end);
  end
  out = [zeros(offset,1); inp2; zeros(len-ss,1)];
end
end