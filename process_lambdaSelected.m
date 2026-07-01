%% process_lambdaSelected: Automatic lambda selection for ECW optimization
% This script selects the best regularization strength (lambda) from
% precomputed optimization results by finding the knee of the
% NLL-vs-lambda curve using the maximum-curvature criterion described in the manuscript.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Optimization results saved by `process_Optimization.m` in `data_Optimization/`
%   - Curve smoothing and basic MATLAB toolboxes
%
% USAGE:
%   Run this script after running the optimization pipeline. It loads
%   `Result_Sub_<id>.mat` files from `data_Optimization/` and selects a
%   best lambda and associated parameter set per subject.
%
% OUTPUTS:
%   - data_Optimization/Best_Lambda_Weights_Strategy.mat : struct of selected
%     lambdas and parameter vectors for each processed subject
%
% SEE ALSO:
%   process_Optimization.m
%
% REFERENCES:
%   See manuscript: Automatic lambda selection and regularization strategy
%   described in the Methods section (Optimization Strategy and Regularization).
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear; clc; close all;

% ===================== 1. Configuration (Control Panel) =====================
subject_ids = [6, 7, 8, 9, 10];
base_dir = 'data_Optimization';
save_filename = 'Best_Lambda_Weights_Strategy.mat';

% ===================== 2. Lambda selection processing =====================

% initialize results storage
All_Best_Results = struct();
fprintf('=== Start automatic lambda selection ===\n\n');

for s_idx = 1:length(subject_ids)
    curr_id = subject_ids(s_idx);
    result_file = fullfile(base_dir, sprintf('Result_Sub_%d.mat', curr_id));
    
    if ~exist(result_file, 'file')
        warning('Result file for subject %d is missing; skipping.', curr_id);
        continue;
    end
    
    load(result_file, 'Subject_Results');
    
    % --- select valid lambda results ---
    % Filter out lambdas for which optimization did not converge.
    valid_mask = arrayfun(@(x) ~isempty(x.best_params), Subject_Results);
    Results = Subject_Results(valid_mask);
    lambdas = [Results.lambda];
    [lambdas, sort_idx] = sort(lambdas); % sort lambda values in ascending order
    Results = Results(sort_idx);
    N = length(lambdas);
    
    % --- extract the curve to evaluate ---
    % Use the total NLL sum for each lambda to form the selection curve.
    nll_vals = arrayfun(@(x) x.best_nll_sum, Results);
    y_raw = nll_vals(:);
    
    % X is lambda, corresponding to the regularization strength.
    x_raw = lambdas(:);
    
    % --- smooth and normalize the curve for curvature analysis ---
    x_log = log10(x_raw + eps);
    y_smooth = smooth(y_raw, 3, 'moving'); % apply a moving average smoother to NLL
    
    x_min = min(x_log); x_max = max(x_log);
    y_min = min(y_smooth); y_max = max(y_smooth);
    
    % normalize both axes to [0,1] so that curvature is scale-invariant
    x_norm = (x_log - x_min) / (x_max - x_min + eps);
    y_norm = (y_smooth - y_min) / (y_max - y_min + eps);
    
    % --- knee detection using the maximum-curvature criterion ---
    % Compute the discrete curvature of the normalized curve using the
    % circumcircle formula for each triplet of adjacent points.
    metric_vals = zeros(N, 1);
    
    for i = 2 : N-1
        p1 = [x_norm(i-1), y_norm(i-1)];
        p2 = [x_norm(i),   y_norm(i)];
        p3 = [x_norm(i+1), y_norm(i+1)];
        
        % Compute the radius-based curvature metric for the three consecutive points
        a = norm(p2 - p3); b = norm(p1 - p3); c = norm(p1 - p2);
        s = (a + b + c) / 2;
        Area = sqrt(max(0, s * (s - a) * (s - b) * (s - c)));
        if Area > 1e-9
            metric_vals(i) = (4 * Area) / (a * b * c);
        end
    end
    
    % Choose the lambda corresponding to the maximum curvature point.
    [~, idx_knee] = max(metric_vals);
    best_lambda = lambdas(idx_knee);

    % --- retrieve the parameter set at the selected lambda ---
    best_entry = Results(idx_knee);
    best_params = best_entry.best_params;
    
    field_name = sprintf('Sub%d', curr_id);
    All_Best_Results.(field_name).id = curr_id;
    All_Best_Results.(field_name).best_lambda = best_lambda;
    All_Best_Results.(field_name).weights = best_params(1:29);
    All_Best_Results.(field_name).prior = best_params(30);
    All_Best_Results.(field_name).motion = best_params(31);
    All_Best_Results.(field_name).full_params = best_params;
    All_Best_Results.(field_name).nll_sum = best_entry.best_nll_sum;
    
    fprintf('   -> Subject %d selected: Best Lambda = %.5f | NLL Sum = %.2f\n', ...
        curr_id, best_lambda, best_entry.best_nll_sum);
end

% --- save the final selected lambda strategy ---
save(fullfile(base_dir, save_filename), 'All_Best_Results');
fprintf('\n>>> All subjects have been processed.\n>>> Saved selection results to: %s\n', fullfile(base_dir, save_filename));