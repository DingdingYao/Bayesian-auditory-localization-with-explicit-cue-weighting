%% Figure 5: Vocoder spectral-resolution experiment with repeated simulations and yao2026 predictions
% This script evaluates the effect of reduced spectral resolution on
% localization performance using the proposed ECW model, the Barumerli 2023
% benchmark, and the Baumgartner 2014 reference.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Auditory Modeling Toolbox (AMT) and its startup script `amt_start`
%   - Result file `data_Optimization/Best_Lambda_Weights_Strategy.mat`
%   - barumerli2023 and baumgartner2014 data available via `amt_load`
%
% USAGE:
%   Run this script after the optimization and lambda selection pipeline.
%   It loads optimized ECW parameters, computes vocoder-based predictions for
%   each channel condition, and generates the Figure 5 manuscript plot.
%
% OUTPUTS:
%   - Figure5.pdf : manuscript-ready figure file
%   - Figure5.mat : cached vocoder experiment results
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m, Figure1.m, Figure2.m, Figure3.m, Figure4.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear all; close all; clc;

if ~exist('amt_start', 'file'), error('Please install the AMT toolbox or add it to the MATLAB path'); end
amt_start;

% ===================== 1. Experiment setup and caching =====================
cache_filename = 'Figure5.mat'; 

if exist(cache_filename, 'file')
    fprintf('\n>>> [Cache] Found cached simulation file %s, loading results...\n', cache_filename);
    load(cache_filename);
else
    fprintf('\n>>> [Cache miss] %s not found, starting full vocoder simulation pipeline...\n', cache_filename);
    
    feature_type = 'pge';
    subject_ids = [6, 7, 8, 9, 10]; 
    nh_ids = {'NH12', 'NH15', 'NH16', 'NH17', 'NH18'}; % participant identifiers matching the behavioral dataset
    num_sbj = length(subject_ids);
    
    % Total_runs controls repetition averaging; if model predictions are deterministic, results are identical across runs
    total_runs = 50; 
    fs = 48000;
    
    fprintf('>>> [1/4] Loading model parameters...\n');
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

    Conditions = {'CL', 'N24', 'N18', 'N12', 'N9', 'N6', 'N3'};
    N_channels = [30, 24, 18, 12, 9, 6, 3]; 
    flow = 300; fhigh = 16000;  

    % ===================== 2. Load the original Goupell human reference data =====================
    fprintf('>>> [2/4] Loading the Goupell vocoder behavioral reference data...\n');
    
    hum_pe_all = NaN(8, length(Conditions)); hum_qe_all = NaN(8, length(Conditions)); hum_gainp_all = NaN(8, length(Conditions));
    part_ids = {'NH12', 'NH15'}; 
    part_row_indices = [];

    for C = 1:length(Conditions)
        data_g = data_goupell2010(Conditions{C});
        num_goupell = length(data_g); 
        
        for g = 1:num_goupell
            mtx_curr = data_g(g).mtx;
            
            % Recover the full-spectrum target directions in the human reference data
            t_sph = [mtx_curr(:,1:2), ones(size(mtx_curr,1),1)];
            % Correct mapping: data_goupell2010 stores response azimuth and elevation in columns 3 and 4
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

    % Initialize the 4 model result matrices
    ecw_pe_pool = NaN(num_sbj, length(Conditions)); ecw_qe_pool = NaN(num_sbj, length(Conditions)); ecw_gainp_pool = NaN(num_sbj, length(Conditions));
    ecw_pe_part = NaN(num_sbj, length(Conditions)); ecw_qe_part = NaN(num_sbj, length(Conditions)); ecw_gainp_part = NaN(num_sbj, length(Conditions));
    
    baru_pe_pool = NaN(num_sbj, length(Conditions)); baru_qe_pool = NaN(num_sbj, length(Conditions)); baru_gainp_pool = NaN(num_sbj, length(Conditions));
    baru_pe_part = NaN(num_sbj, length(Conditions)); baru_qe_part = NaN(num_sbj, length(Conditions)); baru_gainp_part = NaN(num_sbj, length(Conditions));
    
    baum_pe_pool = NaN(num_sbj, length(Conditions)); baum_qe_pool = NaN(num_sbj, length(Conditions)); baum_gainp_pool = NaN(num_sbj, length(Conditions));
    baum_pe_part = NaN(num_sbj, length(Conditions)); baum_qe_part = NaN(num_sbj, length(Conditions)); baum_gainp_part = NaN(num_sbj, length(Conditions));

    % ===================== 3. Run model predictions for each vocoder condition =====================
    fprintf('>>> [3/4] Running vocoder condition predictions (repeat averaging)...\n');
    for i = 1:num_sbj
        curr_id = subject_ids(i); curr_nh = nh_ids{i};
        is_part_subject = (i <= 2); 
        fprintf('\n========== Processing subject %s (SOFA ID: %d) ==========\n', curr_nh, curr_id);
        
        global_full_params = All_Best_Results.(sprintf('Sub%d', curr_id)).full_params;
        summary.alphas = global_full_params(1:29); summary.prior = global_full_params(30); summary.motion = global_full_params(31);
        
        sigma_l = calibrations.sigma(i, pge_idx).values(1); sigma_l2 = calibrations.sigma(i, pge_idx).values(2);
        sigma_mon = calibrations.sigma(i, pge_idx).values(3); sigma_m = calibrations.sigma(i, pge_idx).values(4); sigma_prior = calibrations.sigma(i, pge_idx).values(5);
        
        data_majdak = data_majdak2010('Learn_M');
        sofa = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', data_majdak(curr_id).id));
        
        % --- find the reference front-facing SOFA impulse response h0 for vocoder normalization ---
        raw_sph = barumerli2023_coordinates(sofa).return_positions('spherical');
        dist_to_front = min(abs(raw_sph(:,1)), abs(raw_sph(:,1)-360)) + abs(raw_sph(:,2));
        [~, idx_h0] = min(dist_to_front);
        h0_sofa = squeeze(sofa.Data.IR(idx_h0, 1, :)); 
        
        % --- extract the full-spectrum template representation for the 1500-location grid ---
        [template, ~] = barumerli2023_featureextraction(sofa, feature_type);
        template_feat = [template.itd, template.ild, template.monaural];
        Tmpl = struct(); Tmpl.coords_cart = template.coords.return_positions('cartesian');
        Tmpl.grid_el = template.coords.return_positions('spherical'); Tmpl.grid_el = Tmpl.grid_el(:,2);
        Tmpl.mu = mean(template_feat); Tmpl.std = std(template_feat) + eps;
        Tmpl.features = bsxfun(@rdivide, bsxfun(@minus, template_feat, Tmpl.mu), Tmpl.std);
        
        % --- define the pooled median-lateral region (within б└10бу) from the 1500-location grid ---
        cart_grid = Tmpl.coords_cart;
        latpol_grid = template.coords.return_positions('horizontal-polar');
        idx_pool = find(latpol_grid(:, 1) >= -10 & latpol_grid(:, 1) <= 10); 
        obj_pool = barumerli2023_coordinates(cart_grid(idx_pool, :), 'cartesian');
        pool_cart = obj_pool.return_positions('cartesian'); 
        
        % Load the Baumgartner reference data for the current subject, if available
        s_obj = data_baumgartner2014('pool'); pool_idx = find(strcmp({s_obj.id}, curr_nh));
        if ~isempty(pool_idx), spdtfs_ref = s_obj(pool_idx).Obj; [spdtfs_tmpl, polang_baum] = extractsp(0, spdtfs_ref); end
        
        for C = 1:length(Conditions)
            Cond = Conditions{C}; n_ch = N_channels(C);
            fprintf('  -> Processing condition %s : ', Cond);
            
            sofa_cond = sofa; spdtfs_cond = spdtfs_tmpl;
            
            % --- apply the vocoder to the test SOFA data ---
            if ~strcmp(Cond, 'CL')
                stimPar.SamplingRate = fs; imp = [1; zeros(2^12-1, 1)]; 
                [syncrnfreq, GETtrain] = local_getVocoder('', imp, n_ch, flow, fhigh, 0, 100, stimPar);
                corners = [syncrnfreq(1); syncrnfreq(:,2)];
                
                % Baumgartner SP-DTF frequency-domain representation
                if ~isempty(pool_idx)
                    ref_baum = spdtfs_tmpl; spdtfs_cond = zeros(length(imp), size(ref_baum, 2), 2);
                    for ch = 1:size(ref_baum, 3)
                        for ang = 1:size(ref_baum, 2)
                            spdtfs_cond(:, ang, ch) = local_channelize('', 0.5*ref_baum(:,ang,ch), ref_baum(:,1), imp, n_ch, corners, [], GETtrain, stimPar, 1, 0.01*fs, 0.01*fs);
                        end
                    end
                end
                
                % Use the full-spectrum reference SOFA and 1550 frequency bins to preserve spatial fidelity
                num_total_dirs = size(sofa.Data.IR, 1);
                new_IR = zeros(num_total_dirs, 2, length(imp));
                fprintf('[frequency domain ');
                for dir = 1:num_total_dirs
                    if mod(dir, 300) == 0, fprintf('.'); end 
                    new_IR(dir, 1, :) = local_channelize('', 0.5*squeeze(sofa.Data.IR(dir, 1, :)), h0_sofa, imp, n_ch, corners, [], GETtrain, stimPar, 1, 0.01*fs, 0.01*fs);
                    new_IR(dir, 2, :) = local_channelize('', 0.5*squeeze(sofa.Data.IR(dir, 2, :)), h0_sofa, imp, n_ch, corners, [], GETtrain, stimPar, 1, 0.01*fs, 0.01*fs);
                end
                fprintf('] ');
                sofa_cond.Data.IR = new_IR;
            end
            
            % --- extract the full-spectrum features from the processed SOFA data ---
            [tgt_extraction] = barumerli2023_featureextraction(sofa_cond, feature_type);
            TargetAll_feat = [tgt_extraction.itd, tgt_extraction.ild, tgt_extraction.monaural];
            num_itd = size(tgt_extraction.itd, 2); num_ild = size(tgt_extraction.ild, 2); num_mon = size(tgt_extraction.monaural, 2);
            
            % Pool condition features: use raw target features for yao2026, without additional standardization
            pool_feat_raw = TargetAll_feat(idx_pool, :);
            tgt_pool_ecw = struct('features', pool_feat_raw, 'target_cart', pool_cart);
            tgt_pool_baru = struct('itd', pool_feat_raw(:, 1:num_itd), 'ild', pool_feat_raw(:, num_itd+1:num_itd+num_ild), 'monaural', pool_feat_raw(:, end-num_mon+1:end), 'coords', obj_pool);
            
            % Participant-specific condition: use the original experiment targets for NH12 and NH15
            g_idx = find(strcmp({data_g.id}, curr_nh));
            run_part = is_part_subject && ~isempty(g_idx);
            if run_part
                data_g_temp = data_goupell2010(Cond); 
                g_idx_temp = find(strcmp({data_g_temp.id}, curr_nh));
                mtx_subj = data_g_temp(g_idx_temp).mtx; 
                
                idlat = mtx_subj(:,7) <= 10 & mtx_subj(:,7) >= -10;
                part_az = mtx_subj(idlat, 1); part_el = mtx_subj(idlat, 2); part_pol = mtx_subj(idlat, 6);
                
                % Find the nearest pooled grid directions to the original target locations
                obj_part_real = barumerli2023_coordinates([part_az, part_el, ones(length(part_az),1)], 'spherical');
                part_cart_real = obj_part_real.return_positions('cartesian');
                [~, best_idx_part] = max(part_cart_real * cart_grid', [], 2);
                part_feat_raw = TargetAll_feat(best_idx_part, :);
                obj_part_matched = barumerli2023_coordinates(cart_grid(best_idx_part, :), 'cartesian');
                part_cart_matched = obj_part_matched.return_positions('cartesian');
                
                tgt_part_ecw = struct('features', part_feat_raw, 'target_cart', part_cart_matched);
                tgt_part_baru = struct('itd', part_feat_raw(:, 1:num_itd), 'ild', part_feat_raw(:, num_itd+1:num_itd+num_ild), 'monaural', part_feat_raw(:, end-num_mon+1:end), 'coords', obj_part_matched);
            end
            
            % --- repeat model predictions for averaging ---
            val_ecw_pl_pe=NaN(total_runs,1); val_ecw_pl_qe=NaN(total_runs,1); val_ecw_pl_g=NaN(total_runs,1);
            val_baru_pl_pe=NaN(total_runs,1); val_baru_pl_qe=NaN(total_runs,1); val_baru_pl_g=NaN(total_runs,1);
            val_baum_pl_pe=NaN(total_runs,1); val_baum_pl_qe=NaN(total_runs,1); val_baum_pl_g=NaN(total_runs,1);
            
            val_ecw_pa_pe=NaN(total_runs,1); val_ecw_pa_qe=NaN(total_runs,1); val_ecw_pa_g=NaN(total_runs,1);
            val_baru_pa_pe=NaN(total_runs,1); val_baru_pa_qe=NaN(total_runs,1); val_baru_pa_g=NaN(total_runs,1);
            val_baum_pa_pe=NaN(total_runs,1); val_baum_pa_qe=NaN(total_runs,1); val_baum_pa_g=NaN(total_runs,1);
            
            for s = 1:total_runs
                % ---- pool prediction (pooled group) ----
                m_ecw_pool = yao2026(summary, Tmpl, tgt_pool_ecw);
                res = barumerli2023_metrics(m_ecw_pool, 'middle_metrics');
                val_ecw_pl_pe(s)=res.rmsP; val_ecw_pl_qe(s)=res.querr; val_ecw_pl_g(s)=res.gainP;
                
                m_baru_pool = barumerli2023('template', template, 'target', tgt_pool_baru, 'num_exp', 1, 'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, 'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m, 'sigma_prior', sigma_prior);
                res = barumerli2023_metrics(m_baru_pool, 'middle_metrics');
                val_baru_pl_pe(s)=res.rmsP; val_baru_pl_qe(s)=res.querr; val_baru_pl_g(s)=res.gainP;
                
                if ~isempty(pool_idx)
                    [p_map, rang] = baumgartner2014(spdtfs_cond, spdtfs_tmpl, fs, 'S', s_obj(pool_idx).S, 'lat', 0, 'polsamp', polang_baum);
                    m_v_p = local_baumgartner2014_virtualexp(p_map, polang_baum, rang, 'runs', 1);
                    m_baum_pool = [zeros(size(m_v_p, 1), 4), m_v_p(:, 5:8)]; 
                    res = barumerli2023_metrics(m_baum_pool, 'middle_metrics');
                    val_baum_pl_pe(s)=res.rmsP; val_baum_pl_qe(s)=res.querr; val_baum_pl_g(s)=res.gainP;
                end
                
                % ---- part prediction (participant-specific group) ----
                if run_part
                    m_ecw_part = yao2026(summary, Tmpl, tgt_part_ecw);
                    res = barumerli2023_metrics(m_ecw_part, 'middle_metrics');
                    val_ecw_pa_pe(s)=res.rmsP; val_ecw_pa_qe(s)=res.querr; val_ecw_pa_g(s)=res.gainP;
                    
                    m_baru_part = barumerli2023('template', template, 'target', tgt_part_baru, 'num_exp', 1, 'sigma_itd', sigma_l, 'sigma_ild', sigma_l2, 'sigma_spectral', sigma_mon, 'sigma_motor', sigma_m, 'sigma_prior', sigma_prior);
                    res = barumerli2023_metrics(m_baru_part, 'middle_metrics');
                    val_baru_pa_pe(s)=res.rmsP; val_baru_pa_qe(s)=res.querr; val_baru_pa_g(s)=res.gainP;
                    
                    if ~isempty(pool_idx)
                        % For the three participant-specific target directions, use matched lat=0 responses
                        m_v_pa = local_baumgartner2014_virtualexp(p_map, polang_baum, rang, 'targetset', part_pol, 'runs', 1);
                        m_baum_part = [zeros(size(m_v_pa, 1), 4), m_v_pa(:, 5:8)];
                        res = barumerli2023_metrics(m_baum_part, 'middle_metrics');
                        val_baum_pa_pe(s)=res.rmsP; val_baum_pa_qe(s)=res.querr; val_baum_pa_g(s)=res.gainP;
                    end
                end
            end % end repeat averaging loop
            
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

    fprintf('>>> [4/4] Saving the computed results...\n');
    save(cache_filename, ...
        'ecw_pe_part', 'ecw_qe_part', 'ecw_gainp_part', 'baru_pe_part', 'baru_qe_part', 'baru_gainp_part', 'baum_pe_part', 'baum_qe_part', 'baum_gainp_part', ...
        'ecw_pe_pool', 'ecw_qe_pool', 'ecw_gainp_pool', 'baru_pe_pool', 'baru_qe_pool', 'baru_gainp_pool', 'baum_pe_pool', 'baum_qe_pool', 'baum_gainp_pool', ...
        'hum_pe_all', 'hum_qe_all', 'hum_gainp_all', 'part_row_indices', 'Conditions', 'N_channels');
end

% ===================== 4. Figure 5 rendering (2x3 layout) =====================
fprintf('>>> Rendering the 2x3 figure...\n');

FontSize = 11; MarkerSize = 7; 
% [Plot colors] Match Figure 4 style
Color_Hum = [0.15, 0.15, 0.15];            % Human reference (Actual Human)
Color_My_ECW = [0.0000, 0.3500, 0.6500];   % Proposed model (ECW)
Color_Baru = [0.8500, 0.2000, 0.1500];     % barumerli2023 reference
Color_Baum = [0.4660, 0.6740, 0.1880];     % baumgartner2014 reference

% Line order mapping: Human -> ECW -> Baru -> Baum
dx = 0.12; 
off_hum = -1.5; off_my = -0.5; off_baru = 0.5; off_baum = 1.5;

x_pos = [7, 6, 5, 4, 3, 2, 1]; 
x_ticks = 1:7; x_labels = {'3','6','9','12','18','24','CL'};

% Compute robust quantiles for each condition (handles NaNs safely)
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
fig = figure('Name','Vocoder Test (Part vs Pool)', 'Position',[50 50 1200 520], 'Color','w'); 
left_pad = 0.07; right_pad = 0.02; bot_pad = 0.092; top_pad = 0.05; gap_x = 0.05; gap_y = 0.06;
w_ax = (1 - left_pad - right_pad - 2*gap_x) / 3; h_ax = (1 - bot_pad - top_pad - gap_y) / 2;
y_top = bot_pad + h_ax + gap_y;

% ===================== 4.1 [Row 1: Part] =====================
% Row 1: Participant-specific results
ax1 = axes('Position', [left_pad, y_top, w_ax, h_ax]); hold on;
local_middlebroxplot(ax1, x_pos+off_hum*dx, hum_pe_part_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
local_middlebroxplot(ax1, x_pos+off_my*dx, ecw_pe_p_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax1, x_pos+off_baru*dx, baru_pe_p_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax1, x_pos+off_baum*dx, baum_pe_p_q, 'pd', MarkerSize, Color_Baum, 'w');
set(ax1, 'XLim', [0.5 7.5], 'XTick', x_ticks, 'XTickLabel', [], 'YLim', [25 55], 'FontSize', FontSize);
% Y-axis label for the first row
ylabel('PE (deg)', 'FontSize', 13, 'FontWeight', 'bold'); 
title('Polar Error (PE)', 'FontWeight', 'bold', 'FontSize', 14); grid on; box on;
% Row label for 'Part.'
text(ax1, -0.18, 0.5, 'Part.', 'Units', 'normalized', 'Rotation', 90, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');

ax2 = axes('Position', [left_pad+w_ax+gap_x, y_top, w_ax, h_ax]); hold on;
local_middlebroxplot(ax2, x_pos+off_hum*dx, hum_qe_part_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
local_middlebroxplot(ax2, x_pos+off_my*dx, ecw_qe_p_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax2, x_pos+off_baru*dx, baru_qe_p_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax2, x_pos+off_baum*dx, baum_qe_p_q, 'pd', MarkerSize, Color_Baum, 'w');
set(ax2, 'XLim', [0.5 7.5], 'XTick', x_ticks, 'XTickLabel', [], 'YLim', [0 45], 'FontSize', FontSize);
ylabel('QE (%)', 'FontSize', 13, 'FontWeight', 'bold'); 
title('Quadrant Error (QE)', 'FontWeight', 'bold', 'FontSize', 14); grid on; box on;

ax3 = axes('Position', [left_pad+2*w_ax+2*gap_x, y_top, w_ax, h_ax]); hold on;
local_middlebroxplot(ax3, x_pos+off_hum*dx, hum_g_part_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
local_middlebroxplot(ax3, x_pos+off_my*dx, ecw_g_p_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax3, x_pos+off_baru*dx, baru_g_p_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax3, x_pos+off_baum*dx, baum_g_p_q, 'pd', MarkerSize, Color_Baum, 'w');
set(ax3, 'XLim', [0.5 7.5], 'XTick', x_ticks, 'XTickLabel', [], 'YLim', [-0.2 1.2], 'FontSize', FontSize);
ylabel('Gain', 'FontSize', 13, 'FontWeight', 'bold'); 
title('Polar Gain (gainP)', 'FontWeight', 'bold', 'FontSize', 14); grid on; box on;

% Draw a custom legend on the first axes (so it does not overlap the other panels)
axes(ax1); hold on;
h_l1 = plot(NaN,NaN,'ko-','LineWidth',1.5,'MarkerSize',MarkerSize+1,'MarkerFaceColor',Color_Hum);
h_l2 = plot(NaN,NaN,'s-','Color',Color_My_ECW,'LineWidth',1.5,'MarkerSize',MarkerSize+1,'MarkerFaceColor','w');
h_l3 = plot(NaN,NaN,'s-','Color',Color_Baru,'LineWidth',1.5,'MarkerSize',MarkerSize,'MarkerFaceColor','w');
h_l4 = plot(NaN,NaN,'d-','Color',Color_Baum,'LineWidth',1.5,'MarkerSize',MarkerSize,'MarkerFaceColor','w');
lgd = legend([h_l1, h_l2, h_l3, h_l4], {'Actual Human', 'Proposed method', 'barumerli2023', 'baumgartner2014'}, ...
    'Location', 'southwest', 'FontSize', 11);
set(lgd, 'Box', 'on', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1, 'Color', [1 1 1 0.85]);


% ===================== 4.2 [Row 2: Pool] =====================
ax4 = axes('Position', [left_pad, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax4, x_pos+off_hum*dx, hum_pe_pool_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
local_middlebroxplot(ax4, x_pos+off_my*dx, ecw_pe_pl_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax4, x_pos+off_baru*dx, baru_pe_pl_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax4, x_pos+off_baum*dx, baum_pe_pl_q, 'pd', MarkerSize, Color_Baum, 'w');
set(ax4, 'XLim', [0.5 7.5], 'XTick', x_ticks, 'XTickLabel', x_labels, 'YLim', [25 55], 'FontSize', FontSize);
ylabel('PE (deg)', 'FontSize', 13, 'FontWeight', 'bold'); 
xlabel('Vocoder Channels', 'FontSize', 14, 'FontWeight', 'bold'); grid on; box on;
% Row label for 'Pool' below, with titles on the bottom row
text(ax4, -0.18, 0.5, 'Pool', 'Units', 'normalized', 'Rotation', 90, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');

ax5 = axes('Position', [left_pad+w_ax+gap_x, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax5, x_pos+off_hum*dx, hum_qe_pool_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
local_middlebroxplot(ax5, x_pos+off_my*dx, ecw_qe_pl_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax5, x_pos+off_baru*dx, baru_qe_pl_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax5, x_pos+off_baum*dx, baum_qe_pl_q, 'pd', MarkerSize, Color_Baum, 'w');
set(ax5, 'XLim', [0.5 7.5], 'XTick', x_ticks, 'XTickLabel', x_labels, 'YLim', [0 45], 'FontSize', FontSize);
ylabel('QE (%)', 'FontSize', 13, 'FontWeight', 'bold'); 
xlabel('Vocoder Channels', 'FontSize', 14, 'FontWeight', 'bold'); grid on; box on;

ax6 = axes('Position', [left_pad+2*w_ax+2*gap_x, bot_pad, w_ax, h_ax]); hold on;
local_middlebroxplot(ax6, x_pos+off_hum*dx, hum_g_pool_q, 'ko', MarkerSize+1, Color_Hum, Color_Hum);
local_middlebroxplot(ax6, x_pos+off_my*dx, ecw_g_pl_q, 'ps', MarkerSize+1, Color_My_ECW, 'w');
local_middlebroxplot(ax6, x_pos+off_baru*dx, baru_g_pl_q, 'ps', MarkerSize, Color_Baru, 'w');
local_middlebroxplot(ax6, x_pos+off_baum*dx, baum_g_pl_q, 'pd', MarkerSize, Color_Baum, 'w');
set(ax6, 'XLim', [0.5 7.5], 'XTick', x_ticks, 'XTickLabel', x_labels, 'YLim', [-0.2 1.2], 'FontSize', FontSize);
ylabel('Gain', 'FontSize', 13, 'FontWeight', 'bold'); 
xlabel('Vocoder Channels', 'FontSize', 14, 'FontWeight', 'bold'); grid on; box on;

% ===================== 5. Save figure output =====================
fprintf('>>> Arranging and printing the figure...\n');
set(fig, 'Units', 'Inches'); pos = get(fig, 'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
print(fig, 'Figure5.pdf', '-dpdf', '-painters');
fprintf('>>> Finished. Open Figure5.pdf to review.\n');

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

% --- helper functions end ---
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

if isempty(kv.targetset)
  kv.targetset = tang;
end

nt=length(kv.targetset);
m = nan(nt*kv.runs,9);
m(:,5) = kv.lat;
m(:,6) = repmat(kv.targetset(:),kv.runs,1);
m(:,7) = kv.lat;
if length(tang) > 1
  tangbound = tang(:)+0.5*diff([tang(1)-diff(tang(1:2));tang(:)]);
else
  tangbound = tang;
end
post=zeros(nt,1); 
for ii = 1:nt
  if kv.targetset(ii) > max(tangbound)
    post(ii) = length(tangbound); 
  else
    post(ii) = find(tangbound>=kv.targetset(ii),1);
  end
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
p = p/sum(p);   
c = cumsum(p);
t = max(c)*rand(n,m); 
X = zeros(n,m);
for jj = 1:m
    for ii = 1:n
        X(ii,jj) = find(c >= t(ii,jj) ,1);
    end
end
end


function [syncrnfreq, GETtrain] = local_getVocoder(filename,in,channum,lower,upper,alpha,GaussRate,stimpar)
warning('off')
% channel/timing parameters
srate=stimpar.SamplingRate;
nsamples=length(in); % length of sound record
duration=nsamples/srate; % duration of the signal (in s)
t=0:1/srate:duration;
t=t(1:nsamples);
extendedrange = 0;
if alpha == -1 % Log12ER case
    extendedrange = 1;
    alpha = 0.28;
elseif alpha == 0 % use predefined alphas
  switch channum
    case 3
      alpha=1.42;
    case 6
      alpha=0.67;
    case 9
      alpha=0.45;
    case 12
      alpha=0.33;
    case 18
      alpha=0.22;
    case 24
      alpha=0.17;
    otherwise
      error(['Alpha not predefined for channels number of ' num2str(channu)]);
  end
end

% Synthesis: These are the crossover frequencies that the output signal is mapped to
crossoverfreqs = logspace( log10(lower), log10(upper), channum + 1);
if extendedrange == 1
    crossoverfreqs = [300,396,524,692,915,1209,1597,2110,2788,4200,6400,10000,16000];
end
syncrnfreq(:,1)=crossoverfreqs(1:end-1);
syncrnfreq(:,2)=crossoverfreqs(2:end);
for i=1:channum
    cf(i) = sqrt( syncrnfreq(i,1)*syncrnfreq(i,2) );
end

% Pulse train parameters
Gamma = alpha*cf;
if extendedrange == 1
    Gamma(9) = 1412;
    Gamma(10) = 2200;
    Gamma(11) = 3600;
    Gamma(12) = 6000;
end
N = duration*GaussRate; % number of pulses
fN = floor(N);
Genv=zeros(fN,nsamples);
GETtrain=zeros(channum,nsamples);

% Make pulse trains
for i = 1:channum
    Teff = 1000/Gamma(i);
    if Teff > 3.75
    % if modulation depth is not 100%, make pulse train then modulate
        for n = 1:N
            % delay pulses by half a period so first Gaussian pulse doesn't
            % start at a max
            T = (n-0.5)/N*duration;
            Genv(n,:) = sqrt(Gamma(i)) * exp(-pi*(Gamma(i)*(t-T)).^2);
        end
        Genv_train(i,:) = sum(Genv);
        %modulate carrier
        GETtrain(i,:) = Genv_train(i,:) .* sin(2*pi*cf(i)*t);
        %normalize energy
        Energy(i) = norm(GETtrain(i,:))/sqrt(length(t));
%         Energy(i) = rms(GETtrain(i,:)); % !!!!!!!!!!!!
        GETtrain(i,:) = GETtrain(i,:)/Energy(i);
    else
    % if modulation depth is 100%, make modulated pulses and replicate
        T=(0.5)/N*duration;
        Genv=zeros(fN,nsamples);
        Genv(1,:) = sqrt(Gamma(i)) * exp(-pi*(Gamma(i)*(t-T)).^2) .* sin(2*pi*cf(i)*t - T + pi/4); %!!! (t-T)
        Genv=repmat(Genv(1,:),[fN 1]);
        for n=1:N
            T = round((n)/N*nsamples);
            Genv(n,:)=circshift(Genv(n,:),[1 T-1]);
        end
        GETtrain(i,:) = sum(Genv);
        %normalize energy
        Energy(i) = norm(GETtrain(i,:))/sqrt(length(t));
%         Energy(i) = rms(GETtrain(i,:)); % !!!!!!!!!!!!
        GETtrain(i,:) = GETtrain(i,:)/Energy(i);
    end
end

end

function out=local_channelize(fwavout, h, h0, in, channum, corners, syncrnfreq, ...
                        GETtrain, stimpar, amp, fadein, fadeout)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% *** v1.2.0                                                           %
% GET Vocoder scaled by energy, not envelope                           %
%                                                                      %
% *** v1.1.0                                                           %
% Added Gaussian Envelope Tone (GET) Vocoder                           %
% GET pulse train is generated and passed to this function             %
% M. Goupell, May 2008                                                 %
%                                                                      %
% *** v1.0.0                                                           %
% Modified from ElecRang/matlab/makewav.m  v1.3.1                      %
% To be used with Loca by M. Goupell, Nov 2007                         %
% Now program receives a sound rather than reading a speech file       %
% Rewrote according to the specifications (PM, Jan. 2008)              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% h = hrtf
% h0 = hrtf for reference position
% in = reference noise
% noise = number of channels of noise vectors

srate=stimpar.SamplingRate;
N=length(in);               % length of sound record
d=0.5*srate;                % frequency scalar

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read corner frequencies for channels                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Synthesis: These are the crossover frequencies that the output signal is mapped to
% calculated now in GETVocoder, passed to this function

% Analysis: These are the crossover frequencies that the input signal is subdivided by
if length(corners)<=channum
  error('You need at least one more corner frequency than number of channels');
end
anacrnfreq(:,1)=corners(1:end-1);
anacrnfreq(:,2)=corners(2:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis: Filter Signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inX=fftfilt(h,in);

order=4;     %order of butterworth filter
% out=zeros(1,N);
% filtX=zeros(channum,N);

for i=1:channum
   [b, a]=butter(order, anacrnfreq(i,:)/d);
   out=filter(b, a, inX);  % bandpass-filtering
%    E(i)=norm(out,1)/sqrt(N);
   E(i) = rms(out);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Synthesis                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% each channel
for i=1:channum
    outX(i,:) = GETtrain(i,:) * E(i);
end
% sum me up scotty
out=sum(outX,1);

% for i = 1:channum
%      subplot(channum/3,3,i)
%      plot(outX(i,(1:480)))
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save file                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

out=out*(10^(amp/20))/sqrt(sum(out.^2))*sqrt(sum(in.^2))*sqrt(sum(h.^2))/sqrt(sum(h0.^2));

ii=max(max(abs(out)));
if ii>=1
  error(['Maximum amplitude value is ' num2str(20*log10(ii)) 'dB. Set the HRTF scaling factor lower to avoid clipping']);
end
out=local_FWfade(out,0,fadein,fadeout);
% wavwrite(out,srate,stimpar.Resolution,fwavout);
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