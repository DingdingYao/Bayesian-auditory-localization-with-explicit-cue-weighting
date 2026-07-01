%% Figure 1: Lambda selection and plot generation
% This script selects the optimal regularization strength (lambda)
% using the maximum-curvature criterion described in the manuscript.
% It then generates the publication-ready selection-curve plot used in Figure 1.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Optimization results produced by `process_Optimization.m` saved in
%     the `data_Optimization/` directory (files named `Result_Sub_<id>.mat`)
%   - Basic MATLAB plotting utilities (exportgraphics or print for PDF)
%
% USAGE:
%   Run this script after running the optimization pipeline. The script
%   loads per-subject `Subject_Results`, computes the knee of the
%   NLL-vs-lambda curve using the maximum-curvature criterion, selects the corresponding
%   parameter vector, and exports `Figure1.pdf`.
%
% OUTPUTS:
%   - Figure1.pdf : publication-ready selection-curve figure
%   - data_Optimization/Best_Lambda_Weights_Strategy.mat : selected lambda results (if saved)
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear; clc; close all;

% ===================== 1. Configuration (Control Panel) =====================

% Data selection and IO settings
subject_id = 6; % listener ID for the figure
nh_id = 'NH12'; % matching subject identifier
base_dir = 'data_Optimization'; % directory containing optimization results
save_filename = 'Best_Lambda_Weights_Strategy.mat';

% ===================== 2. Initialize results storage =====================

% Initialize results storage
result_file = fullfile(base_dir, sprintf('Result_Sub_%d.mat', subject_id));

if ~exist(result_file, 'file')
    warning('Result file for subject %d is missing; skipping.', subject_id);
    error('Aborting: result file not found.');
end

load(result_file, 'Subject_Results');

% --- select successful lambda results ---
valid_mask = arrayfun(@(x) ~isempty(x.best_params), Subject_Results);
Results = Subject_Results(valid_mask);
lambdas = [Results.lambda];
[lambdas, sort_idx] = sort(lambdas);
Results = Results(sort_idx);
N = length(lambdas);

% --- prepare the selection curve ---
% Y is the total NLL sum for each fitted lambda.
nll_vals = arrayfun(@(x) x.best_nll_sum, Results);
y_raw = nll_vals(:);

% X is lambda, which will be plotted on a log scale.
x_raw = lambdas(:);
x_label_str = '\lambda';

% --- smooth and normalize the curve ---
x_log = log10(x_raw + eps);
y_smooth = smooth(y_raw, 3, 'moving');

x_min = min(x_log); x_max = max(x_log);
y_min = min(y_smooth); y_max = max(y_smooth);

% normalize both axes to [0,1] for scale-invariant curvature computation.
x_norm = (x_log - x_min) / (x_max - x_min + eps);
y_norm = (y_smooth - y_min) / (y_max - y_min + eps);

% --- knee detection using the maximum-curvature criterion ---
% Compute discrete curvature for each consecutive triplet of points on the
% normalized curve, then select the point with maximum curvature.
metric_vals = zeros(N, 1);
for i = 2 : N-1
    p1 = [x_norm(i-1), y_norm(i-1)];
    p2 = [x_norm(i),   y_norm(i)];
    p3 = [x_norm(i+1), y_norm(i+1)];
    a = norm(p2 - p3); b = norm(p1 - p3); c = norm(p1 - p2);
    s = (a + b + c) / 2;
    Area = sqrt(max(0, s * (s - a) * (s - b) * (s - c)));
    if Area > 1e-9, metric_vals(i) = (4 * Area) / (a * b * c); end
end
metric_name = 'Curvature Value';

[~, idx_knee] = max(metric_vals);
best_lambda = lambdas(idx_knee);
metric_norm = (metric_vals - min(metric_vals)) / (max(metric_vals) - min(metric_vals) + eps);

% --- extract the selected parameter set ---
best_entry = Results(idx_knee);
best_params = best_entry.best_params;


% log progress and report selected lambda and NLL.
fprintf('   -> Subject %d (%s): Best Lambda = %.5f | NLL Sum = %.2f\n', subject_id, nh_id, best_lambda, best_entry.best_nll_sum);

% ===================== 3. Figure 1: Selection curve visualization =====================
% Use a full-width figure and no padding so the plot is publication-ready.
fig2 = figure('Name', 'Selection Curve', 'Position', [100, 100, 1100, 710], 'Color', 'w');
t = tiledlayout(1, 1, 'Padding', 'none');
ax = nexttile;
hold(ax, 'on');

% --- plot raw and smoothed NLL on the left y-axis ---
yyaxis left;
set(ax, 'YColor', 'k');
knee_x = x_norm(idx_knee); knee_y = y_norm(idx_knee);

y_raw_norm = (y_raw - y_min) / (y_max - y_min + eps);
plot(ax, x_norm, y_raw_norm, '-o', 'Color', [0.6 0.8 0.9], 'LineWidth', 2.5, ...
    'MarkerSize', 10, 'MarkerFaceColor', 'w', 'DisplayName', 'Raw NLL');
plot(ax, x_norm, y_norm, '-', 'Color', [0 0.4470 0.7410], 'LineWidth', 5.0, ...
    'DisplayName', 'Smoothed NLL');
plot(ax, knee_x, knee_y, 'p', 'MarkerSize', 25, 'MarkerFaceColor', 'r', ...
    'MarkerEdgeColor', 'r', 'DisplayName', sprintf('Optimal Point (\\lambda=%.4f)', best_lambda));
ylabel('Negative Log-Likelihood (Normalized)', 'FontSize', 28, 'FontWeight', 'bold');
axis(ax, [-0.05 1.05 -0.05 1.05]);

% --- plot the curvature metric on the right y-axis ---
yyaxis right;
set(ax, 'YColor', [0.8500, 0.3250, 0.0980]);
plot(ax, x_norm, metric_norm, '--', 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 4.0, ...
    'DisplayName', metric_name);
ylabel(metric_name, 'FontSize', 28, 'FontWeight', 'bold');
axis(ax, [-0.05 1.05 -0.05 1.35]);

% Draw a vertical line at the chosen knee.
line([knee_x, knee_x], [-0.05, 1.35], 'Color', [0.6 0.6 0.6], 'LineStyle', '-.', 'LineWidth', 2.5, 'HandleVisibility', 'off');

% Format the axes and labels for publication-ready appearance.
set(ax, 'FontName', 'Arial', 'FontSize', 24, 'LineWidth', 2.5);
num_ticks = 6;
tick_pos = linspace(0, 1, num_ticks);
tick_val_log = x_min + tick_pos * (x_max - x_min);
tick_labels = arrayfun(@(v) sprintf('10^{%.1f}', v), tick_val_log, 'UniformOutput', false);
set(ax, 'XTick', tick_pos, 'XTickLabel', tick_labels);
xlabel(sprintf('%s (Log Scale)', x_label_str), 'FontSize', 28, 'FontWeight', 'bold');
grid on; set(ax, 'GridAlpha', 0.25); box on;

lgd = legend('Location', 'northwest');
set(lgd, 'FontSize', 22, 'EdgeColor', 'k', 'LineWidth', 2.0, 'Color', 'w');

% Export the figure to PDF using vector graphics if possible.
save_pdf = 'Figure1.pdf';
fprintf('>>> Exporting Figure 1 to PDF...\n');
try
    exportgraphics(fig2, save_pdf, 'ContentType', 'vector', 'BackgroundColor', 'w');
    fprintf('>>> Export succeeded.\n>>> File path: %s\n', save_pdf);
catch
    warning('exportgraphics failed; falling back to print for PDF export.');
    pos = fig2.Position;
    set(fig2, 'PaperUnits', 'points');
    set(fig2, 'PaperPosition', [0 0 pos(3) pos(4)]);
    set(fig2, 'PaperSize', [pos(3) pos(4)]);
    print(fig2, save_pdf, '-dpdf', '-painters');
end
