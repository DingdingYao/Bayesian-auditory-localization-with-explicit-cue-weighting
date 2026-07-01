%% process_optimization: ECW model parameter optimization
% This script loads Majdak et al. (2010) behavioral data, extracts HRTF
% features with barumerli2023 routines, normalizes template/target cues, and
% fits ECW model parameters using BADS with L2 regularization.
%
% REQUIRES:
%   - MATLAB R2019b or later
%   - Auditory Modeling Toolbox 1.6.0
%   - BADS optimizer
%
% USAGE:
%   Run this script directly. It saves optimization results to
%   `data_Optimization/`.
%
% OUTPUTS:
%   - data_Optimization/Result_Sub_<id>.mat
%
% SEE ALSO:
%   amt_start, amt_load, bads, data_majdak2010,
%   barumerli2023_coordinates, barumerli2023_featureextraction
%
% REFERENCES:
%   Explicit cue weighting in a Bayesian model of auditory localization:
%   Quantifying the relative contributions of binaural and spectral cues.
%   The Journal of the Acoustical Society of America.
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear all; close all; clc;

% ===================== 1. Configuration =====================
feature_type = 'pge'; % feature type for HRTF extraction
subject_ids = 6:10; % Batch subjects for analysis

% List of regularization strengths for alpha weights
lambda_list = logspace(-5, 0, 16); % from 1e-5 to 1 with 16 values on a log scale

% Number of optimization repeats per lambda
num_repeats = 10;
% ==============================================

if ~exist('amt_start', 'file'), error('Auditory Modeling Toolbox (AMT) is required'); end
amt_start;

% Output directory for optimization results
filename = 'data_Optimization';
if ~exist(filename, 'dir'), mkdir(filename); end

amt_disp('Loading behavioral data from Majdak et al. 2010')
full_data_table = data_majdak2010('Learn_M'); 

%% ===================== 2. Main analysis loop =====================
for s_idx = 1:length(subject_ids)
    current_id = subject_ids(s_idx);
    result_file = sprintf('%s/Result_Sub_%d.mat', filename, current_id);
    
    % Check if result file already exists
    if exist(result_file, 'file')
        fprintf('\n[Found] Existing result file: %s\n', result_file);
        fprintf('>>> WARNING: The optimization is VERY time-consuming (hours per subject).\n');
        reply = upper(input('    Press [R] to recompute, or any other key to skip and keep existing: ', 's'));
        if ~strcmp(reply, 'R')
            fprintf('    Skipping subject %d, using existing file.\n\n', current_id);
            continue;
        end
        fprintf('    Recomputing subject %d...\n', current_id);
    end
    
    fprintf('\n\n################################################\n');
    fprintf('Processing subject ID %d (%d/%d)\n', current_id, s_idx, length(subject_ids));
    fprintf('################################################\n');
    
    data_majdak = full_data_table(current_id);
    
    % Behavioral data preprocessing
    targets_deg = data_majdak.mtx(:, 1:2);    % azimuth and elevation in degrees
    responses_deg = data_majdak.mtx(:, 3:4);  % azimuth and elevation in degrees
    
    % Convert target and response directions from spherical angles to Cartesian.
    obj_target = barumerli2023_coordinates([targets_deg, ones(size(targets_deg,1), 1)], 'spherical');
    obj_resp = barumerli2023_coordinates([responses_deg, ones(size(responses_deg,1), 1)], 'spherical');
    
    Data.targets_Cart = obj_target.return_positions('cartesian');
    Data.responses_Cart = obj_resp.return_positions('cartesian');
    
    % Density compensation weights
    % Weight trials to compensate for non-uniform target sampling in the dataset.
    fprintf('   -> Calculating spatial density weights (Gaussian KDE)...\n');
    N_trials = size(Data.targets_Cart, 1);
    
    sigma_deg = 15; 
    sigma_rad = deg2rad(sigma_deg);

    dots = Data.targets_Cart * Data.targets_Cart';
    dots = max(min(dots, 1), -1); % clamp numerical values into [-1,1]
    dists_rad = acos(dots);
    
    densities = sum(exp(-(dists_rad.^2) / (2 * sigma_rad^2)), 2);
    raw_weights = 1 ./ (densities + eps);
    Data.trial_weights = raw_weights / sum(raw_weights) * N_trials; % normalize weights to sum to N_trials
    
    tgt_lat_pol = obj_target.return_positions('horizontal-polar');
    Data.targets_Lat = tgt_lat_pol(:, 1);
    Data.targets_Pol = tgt_lat_pol(:, 2);
    
    % Template and target feature extraction 
    try
        sofa_obj = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', data_majdak.id));
    catch
        error('Error loading SOFA');
    end
    
    % Extract HRTF features for the full template grid.
    [template, ~] = barumerli2023_featureextraction(sofa_obj, feature_type);
    template_feat = [template.itd, template.ild, template.monaural]; % 1500x56 matrix of features for template
    Template_Cart = template.coords.return_positions('cartesian');
    Template_Sph = template.coords.return_positions('spherical');
    Template_lat_pol = template.coords.return_positions('horizontal-polar');
    
    Template.grid_lat = Template_lat_pol(:, 1);
    Template.grid_pol = Template_lat_pol(:, 2);
    Template.grid_el = Template_Sph(:, 2);
    Template.coords_cart = Template_Cart;
    Template.CosDistMatrix = Template_Cart * Template_Cart';
    
    % Extract target feature templates for the observed stimulus directions.
    [target_extraction] = barumerli2023_featureextraction(sofa_obj, feature_type, 'target');
    TargetTemplate_feat = [target_extraction.itd, target_extraction.ild, target_extraction.monaural]; 
    Target_Cart = target_extraction.coords.return_positions('cartesian');
    
    % Map each behavioral target direction to the nearest extracted target template by cosine similarity.
    N = size(Data.targets_Cart, 1);
    target_feat = zeros(N, 56);
    CosSim_S1 = Data.targets_Cart * Target_Cart';
    [~, best_idx_S1] = max(CosSim_S1, [], 2);
    for i = 1:N
        target_feat(i, :) = TargetTemplate_feat(best_idx_S1(i), :);
    end
    
    % Standardize both templates and target features using template statistics.
    mu_Template = mean(template_feat);
    std_Template = std(template_feat) + eps;
    Template.features = (template_feat - mu_Template) ./ std_Template;
    Target.features = (target_feat - mu_Template) ./ std_Template;

    % Weight optimization using BADS and regularization grid search
    num_feats = 29;
    lb  = [zeros(1, num_feats)+1e-4,   1.0,   0.1];
    ub  = [ones(1, num_feats)*50,     60.0,  50.0];
    plb = [ones(1, num_feats)*0.01,   5.0,   1.0];
    pub = [ones(1, num_feats)*5.0,    30.0,  25.0];
    
    options = bads('defaults');
    options.MaxFunEvals = 20000;
    options.Display = 'off';
    options.UncertaintyHandling = 0;
    
    % Search across lambda values and random restarts to reduce local optima.
    nLam = length(lambda_list);
    nRep = num_repeats;
    total_combos = nLam * nRep;
    
    combo_matrix = zeros(total_combos, 2);
    combo_counter = 0;
    for l = 1:nLam
        for r = 1:nRep
            combo_counter = combo_counter + 1;
            combo_matrix(combo_counter, :) = [l, r];
        end
    end
    
    search_history = cell(total_combos, 1);
    for c_idx = 1:total_combos
        l_idx = combo_matrix(c_idx, 1);
        r_idx = combo_matrix(c_idx, 2);
        curr_lambda = lambda_list(l_idx);
        fprintf('   Progress %d/%d: lambda %.5f repeat %d\n', c_idx, total_combos, curr_lambda, r_idx);

        % Define the objective function for the current lambda value.
        obj_fun = @(theta) joint_loss_function(theta, Template, Target, Data, curr_lambda);
        
        x0_rand = plb + (pub - plb) .* rand(size(plb));
        
        curr_output = struct();
        curr_output.lambda_idx = l_idx;
        curr_output.run_idx = r_idx;
        
        try
            [params_curr, nll_curr] = bads(obj_fun, x0_rand, lb, ub, plb, pub, [], options);
            curr_output.params = params_curr;
            curr_output.loss = nll_curr;
        catch ME
            fprintf('Run %d (lambda %.5f, repeat %d) failed: %s\n', c_idx, curr_lambda, r_idx, ME.message);
            curr_output.loss = NaN;
            curr_output.params = [];
        end
        search_history{c_idx} = curr_output;
        
        % Clear intermediate variables to prevent memory accumulation
        clear obj_fun x0_rand curr_output;
        
        % Allow MATLAB to process pending events periodically
        if mod(c_idx, 5) == 0
            drawnow limitrate;
        end
    end
    
    fprintf('>>> Grid search finished; summarizing results...\n');
    
    % Summarize results across repeats for each lambda
    Subject_Results = repmat(struct('id', [], 'lambda', [], 'best_params', [], ...
        'best_nll_sum', [], 'run_history', [], 'fc', [], 'data_mtx', []), nLam, 1);
    
    for l_idx = 1:nLam
        curr_lambda = lambda_list(l_idx);
        
        % Collect the best run for the current lambda value.
        run_history = struct('params', [], 'loss', [], 'nll_sum', []);
        best_nll_local = inf;
        best_params_local = [];
        best_nll_sum_local = inf;
        
        for c_idx = 1:total_combos
            if search_history{c_idx}.lambda_idx == l_idx
                r = search_history{c_idx}.run_idx;
                if ~isempty(search_history{c_idx}.params)
                    p_curr = search_history{c_idx}.params;
                    loss_curr = search_history{c_idx}.loss;
                    
                    % Recover the unregularized NLL sum from the reported loss.
                    alphas = p_curr(1:29);
                    l2_penalty = curr_lambda * 0.5 * sum(alphas.^2);
                    avg_nll_curr = loss_curr - l2_penalty;
                    nll_sum_curr = avg_nll_curr * N_trials;
                    
                    run_history(r).params = p_curr;
                    run_history(r).loss = loss_curr;
                    run_history(r).nll_sum = nll_sum_curr;
                    
                    if loss_curr < best_nll_local
                        best_nll_local = loss_curr;
                        best_params_local = p_curr;
                        best_nll_sum_local = nll_sum_curr;
                    end
                else
                    run_history(r).loss = NaN;
                    run_history(r).params = [];
                    run_history(r).nll_sum = NaN;
                end
            end
        end
        
        if isempty(best_params_local)
            warning('Lambda %.5f had no successful optimization runs.', curr_lambda);
            continue;
        end
        
        Subject_Results(l_idx).id = current_id;
        Subject_Results(l_idx).lambda = curr_lambda;
        Subject_Results(l_idx).best_params = best_params_local;
        Subject_Results(l_idx).best_nll_sum = best_nll_sum_local;
        Subject_Results(l_idx).run_history = run_history;
        Subject_Results(l_idx).fc = template.fc;
        Subject_Results(l_idx).data_mtx = data_majdak.mtx;
        
        fprintf('   [Result] Lambda %.5f | Best Loss: %.4f | Best NLL Sum: %.2f\n', ...
            curr_lambda, best_nll_local, best_nll_sum_local);
    end
    
    save(sprintf('%s/Result_Sub_%d.mat', filename, current_id), 'Subject_Results');
end
fprintf('All done\n');


%% === optimization objective function ===

% joint_loss_function evaluates the regularized negative log-likelihood for a
% candidate parameter vector [alphas, prior, motion].
function total_loss = joint_loss_function(theta, Template, Target, Data, lambda)
alphas = reshape(theta(1:29), 1, 29); % feature weights for the 29 cues
s_prior = theta(30); s_motion = theta(31); % prior width and motion smoothing parameters
X = Target.features; Mu = Template.features; 
N_trials = size(Target.features,1);

% 1. weighted Gaussian likelihood with cue precision parameters
% Approximate the Gaussian likelihood using a diagonal precision matrix
% and feature-specific weights alpha for the template/target features
log_det_term = 0.5 * sum(log(alphas + eps)); 
% replicate alpha weights for binaural cues (same weight for left/right ear)
sqrt_alphas = sqrt([alphas alphas(3:end)]);

% Scale target/template features by their precision weights
X_w = bsxfun(@times, X, sqrt_alphas);
Mu_w = bsxfun(@times, Mu, sqrt_alphas);
X_sq = sum(X_w.^2, 2);
Mu_sq = sum(Mu_w.^2, 2)';

% Weighted squared distance between each target and template
w_dist_matrix = bsxfun(@plus, X_sq, Mu_sq) - 2 * (X_w * Mu_w');
w_dist_matrix = max(w_dist_matrix, 0); % avoid tiny negative values from numerical error

% Compute the log-likelihood 
LogLikelihood = -0.5 * w_dist_matrix + log_det_term;
Prob_L = exp(LogLikelihood - max(LogLikelihood, [], 2)); % subtract max for numerical stability

% Prior distribution over elevation
Prior_dist = reshape(exp(-0.5 * (Template.grid_el / s_prior).^2), 1, size(Mu, 1));

% Posterior over template locations after applying the prior
Prob_Post = Prob_L .* (Prior_dist / (sum(Prior_dist)+eps));

% 2. approximate motor-noise smoothing by convolving the posterior with a
% von Mises-Fisher-like kernel defined over the template cosine-distance matrix.
kappa = 1 / (deg2rad(s_motion)^2 + eps);
Kernel = exp(kappa * Template.CosDistMatrix);
Kernel = Kernel ./ (sum(Kernel, 2) + eps);

Prob_Final = (Prob_Post * Kernel);
Prob_Final = Prob_Final ./ (sum(Prob_Final, 2) + eps); % NxM

% 3. compute weighted negative log-likelihood of the observed responses
Y_cart = Data.responses_Cart;
All_CosSims = Y_cart * Template.coords_cart';

% Find the best-matching template for each response
[~, best_indices] = max(All_CosSims, [], 2);
linear_ind = sub2ind(size(Prob_Final), (1:N_trials)', best_indices);
likelihoods = 0.95 * Prob_Final(linear_ind) + 0.05 * (1/1500); % robustify the likelihood by mixing the model probability with a small uniform floor.
W = Data.trial_weights;
% Weighted loss: -sum(weights .* log(likelihoods))
nll_sum = -sum( W .* log(likelihoods) );
avg_nll = nll_sum / N_trials;

% 4. L2 regularization on the alpha weights
L2_term = sum(alphas.^2);
l2_penalty = lambda * 0.5 * L2_term;

total_loss = avg_nll + l2_penalty;

if isnan(total_loss) || isinf(total_loss), total_loss = 1e9; end
end