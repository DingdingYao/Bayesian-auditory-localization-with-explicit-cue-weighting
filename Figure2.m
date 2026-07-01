%% Figure 2: individual and average cue weights
% This script loads the selected lambda and weight strategy produced by the
% automatic lambda-selection pipeline and generates panels showing
% listener-specific binaural and spectral cue weights as well as across-
% location spectral variability.
%
% REQUIRES:
%   - MATLAB R2019 or later
%   - Auditory Modeling Toolbox (AMT) and its startup script `amt_start`
%   - Results file `data_Optimization/Best_Lambda_Weights_Strategy.mat`
%       (expected to contain the solved parameter sets; if missing, run
%        the optimization + selection pipeline as described below)
%
% USAGE:
%   If you have already have the solved parameter file in
%   `data_Optimization/Best_Lambda_Weights_Strategy.mat`, simply run this
%   script to generate the figures. Otherwise, run `process_lambdaSelected.m`
%   (or the optimization pipeline) first to produce the results, then run
%   this script to visualize the selected weights.
%
% OUTPUTS:
%   - Figure panels displayed and optionally exported by the script.
%
% SEE ALSO:
%   process_Optimization.m, process_lambdaSelected.m, Figure1.m
%
% AUTHOR:
%   Dingding Yao - yaodingding(at)hccl.ioa.ac.cn
%   June 2026

clear; clc; close all;

% Ensure the Auditory Modeling Toolbox startup script is available.
if ~exist('amt_start', 'file')
    error('Auditory Modeling Toolbox startup script not found.');
end
amt_start;

% ===================== 1. Load optimization results =====================
base_dir = 'data_Optimization'; 
mat_file = fullfile(base_dir, 'Best_Lambda_Weights_Strategy.mat');

if ~exist(mat_file, 'file')
    error('Required optimization result file not found: %s', mat_file);
end

fprintf('>>> Loading selected lambda and weight strategy...\n');
load(mat_file);
subject_fields = fieldnames(All_Best_Results);
num_subs = length(subject_fields);

% Frequencies corresponding to the 27 spectral weight dimensions (center frequencies of the gammatone filters)
freqs = [782.2, 897.5, 1025.9, 1169.0, 1328.4, 1505.9, 1703.68, ...
         1924.0, 2169.4, 2442.8, 2747.38, 3086.65, 3464.59, 3885.61, ...
         4354.61, 4877.08, 5459.09, 6107.44, 6829.68, 7634.25, 8530.52, ...
         9528.95, 10641.18, 11880.18, 13260.40, 14797.94, 16510.73];

% Preallocate data arrays for selected weights and fitted parameters.
weights_binaural = zeros(num_subs, 2);
weights_psg      = zeros(num_subs, 27);
params_deg       = zeros(num_subs, 2);
dtf_stds         = zeros(num_subs, 27); 

real_sub_labels  = {'NH12', 'NH15', 'NH16', 'NH17', 'NH18'};

fprintf('>>> Extracting selected weights and HRTF spectral variability...\n');
for i = 1:num_subs
    sub_data = All_Best_Results.(subject_fields{i});
    weights_binaural(i, :) = sub_data.weights(1:2);
    weights_psg(i, :)      = sub_data.weights(3:29);
    params_deg(i, :)       = [sub_data.prior, sub_data.motion];
    
    % --- load subject-specific SOFA and compute PSG spectral standard deviation ---
    try
        sofa_obj = amt_load('barumerli2023', sprintf('ARI_%s_hrtf_M_dtf 256.sofa', real_sub_labels{i}));
        [template, ~] = barumerli2023_featureextraction(sofa_obj, 'pge');
        feat_dim = size(template.monaural, 2);
        if feat_dim == 54
            std_left  = std(template.monaural(:, 1:27), 0, 1);
            std_right = std(template.monaural(:, 28:54), 0, 1);
            dtf_stds(i, :) = mean([std_left; std_right], 1);
        elseif feat_dim == 27
            dtf_stds(i, :) = std(template.monaural, 0, 1);
        else
            dtf_stds(i, :) = std(template.monaural(:, 1:27), 0, 1);
        end
    catch ME
        warning('Failed to load SOFA data for %s: %s', real_sub_labels{i}, ME.message);
    end
end

% Compute across-listener means and standard deviations for plotting.
mean_binaural = mean(weights_binaural, 1); std_binaural = std(weights_binaural, 0, 1);
mean_psg      = mean(weights_psg, 1);      std_psg      = std(weights_psg, 0, 1);
mean_params   = mean(params_deg, 1);       std_params   = std(params_deg, 0, 1);
mean_dtf_std  = mean(dtf_stds, 1);         std_dtf_std  = std(dtf_stds, 0, 1);

% ===================== plot limits and display settings =====================
max_binaural = max([weights_binaural(:); (mean_binaural + std_binaural)']) * 1.1; 
max_psg_val = 0.8; % spectral-weight axis maximum
max_dtf_val = 1.6; % spectral SD axis maximum

yticks_psg = 0:0.2:max_psg_val;
yticks_dtf = 0:0.4:max_dtf_val;

% Plot colors
colors = [
    0.85 0.33 0.10; 0.00 0.45 0.74; 0.47 0.67 0.19; 
    0.49 0.18 0.56; 0.30 0.75 0.93
];
avg_color = [0.2 0.2 0.2]; 
dtf_color = [0.45 0.45 0.45]; 

% ===================== 2. Plot individual and average results =====================
fprintf('>>> Generating Figure 2 panels...\n');
fig = figure('Position', [50, 50, 1100, 720], 'Color', 'w');
t = tiledlayout(6, 5, 'TileSpacing', 'compact', 'Padding', 'none');

% ===================== Top 5 rows: individual listener results =====================
for i = 1:num_subs
    % --- left panel: binaural weights for ITD and ILD ---
    ax_bin = nexttile((i-1)*5 + 1, [1 1]);
    
    % Disable clipping to preserve marker shapes and PDF appearance.
    set(ax_bin, 'Clipping', 'off'); 
    
    stem([1, 2], weights_binaural(i, :), 'filled', 'LineStyle', '-', 'Color', colors(i,:), 'LineWidth', 2.5, 'MarkerSize', 8);
    ylim([0, max_binaural]); xlim([0.5, 2.5]);
    set(ax_bin, 'XTick', [1, 2], 'XTickLabel', {'', ''}, 'FontSize', 14, 'FontName', 'Arial');
    ylabel(sprintf('%s\nBin. Weight', real_sub_labels{i}), 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
    grid on; set(ax_bin, 'GridAlpha', 0.15); box on;
    
    % --- right panel: PSG spectral weights vs across-location variability ---
    ax_psg = nexttile((i-1)*5 + 2, [1 4]);
    set(ax_psg, 'XScale', 'log', 'FontSize', 14, 'FontName', 'Arial');
    
    % Disable clipping so the markers and fill regions render cleanly.
    set(ax_psg, 'Clipping', 'off'); 
    
    % left axis: spectral weights
    yyaxis left
    hold on;
    baseColor = colors(i,:); alphaVal = 0.15; bgColor = [1 1 1]; 
    shadingColor = alphaVal * baseColor + (1 - alphaVal) * bgColor;
    fill([freqs, fliplr(freqs)], [weights_psg(i, :), zeros(1, 27)], shadingColor, 'FaceAlpha', 1, 'EdgeColor', 'none'); 
    
    stem(freqs, weights_psg(i, :), 'filled', 'LineStyle', '-', 'Color', colors(i,:), 'LineWidth', 1.8, 'MarkerSize', 6);
    ylim([0, max_psg_val]);
    yticks(yticks_psg); 
    ylabel('Spec. Weight', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]); 
    set(ax_psg, 'YColor', [0.15 0.15 0.15]);
    
    % right axis: across-location spectral variability
    yyaxis right
    plot(freqs, dtf_stds(i, :), 'LineStyle', '--', 'LineWidth', 2.2, 'Color', colors(i,:));
    ylim([0, max_dtf_val]);
    yticks(yticks_dtf); 
    ylabel('Spec. SD', 'FontSize', 14, 'FontWeight', 'bold', 'Color', dtf_color);
    set(ax_psg, 'YColor', dtf_color); 
    
    xlim([700, 18000]);
    xticks([1000, 2000, 4000, 8000, 16000]);
    set(ax_psg, 'XTickLabel', []);
    grid on; set(ax_psg, 'GridAlpha', 0.15); box on;
    set(ax_psg, 'Layer', 'top');
    
    str_prior = sprintf('%.1f', params_deg(i, 1)); str_motion = sprintf('%.1f', params_deg(i, 2));
    str_anno = sprintf('$\\sigma_{\\mathrm{prior}}$: %s$^\\circ$, $\\sigma_{\\mathrm{motion}}$: %s$^\\circ$', str_prior, str_motion);
    text(0.015, 0.88, str_anno, 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
         'FontSize', 14, 'Interpreter', 'latex', 'BackgroundColor', [1 1 1 0.85], 'EdgeColor', [0.3 0.3 0.3], 'Margin', 3);
end

% ===================== Bottom row: group average with ±1 SD error bars =====================
i = 6;
ax_bin_avg = nexttile((i-1)*5 + 1, [1 1]);
hold on;
% Disable clipping for the average weight axes
set(ax_bin_avg, 'Clipping', 'off');

err_neg_bin = min(mean_binaural, std_binaural);
errorbar([1, 2], mean_binaural, err_neg_bin, std_binaural, 'Color', avg_color, 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 6);
stem([1, 2], mean_binaural, 'filled', 'LineStyle', '-', 'Color', avg_color, 'LineWidth', 2.5, 'MarkerSize', 8);

ylim([0, max_binaural]); xlim([0.5, 2.5]);
set(ax_bin_avg, 'XTick', [1, 2], 'XTickLabel', {'ITD', 'ILD'}, 'FontSize', 14, 'FontName', 'Arial');
ylabel(sprintf('Average\nBin. Weight'), 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
grid on; set(ax_bin_avg, 'GridAlpha', 0.15); box on;

ax_psg_avg = nexttile((i-1)*5 + 2, [1 4]);
set(ax_psg_avg, 'XScale', 'log', 'FontSize', 14, 'FontName', 'Arial');

% Disable clipping for the average spectral panel
set(ax_psg_avg, 'Clipping', 'off');

% left axis: average spectral weights
yyaxis left
hold on;
shadingColor_avg = 0.15 * avg_color + 0.85 * [1 1 1];
fill([freqs, fliplr(freqs)], [mean_psg, zeros(1, 27)], shadingColor_avg, 'FaceAlpha', 1, 'EdgeColor', 'none'); 

err_neg_psg = min(mean_psg, std_psg);
% Plot average spectral weights with ±1 SD error bars
errorbar(freqs, mean_psg, err_neg_psg, std_psg, 'Color', avg_color, 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 4);
stem(freqs, mean_psg, 'filled', 'LineStyle', '-', 'Color', avg_color, 'LineWidth', 1.8, 'MarkerSize', 6);

ylim([0, max_psg_val]);
yticks(yticks_psg);
ylabel('Spec. Weight', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]); 
set(ax_psg_avg, 'YColor', [0.15 0.15 0.15]);

% right axis: average across-location spectral variability
yyaxis right
err_neg_dtf = min(mean_dtf_std, std_dtf_std);
errorbar(freqs, mean_dtf_std, err_neg_dtf, std_dtf_std, 'Color', dtf_color, 'LineStyle', 'none', 'LineWidth', 1, 'CapSize', 4);
plot(freqs, mean_dtf_std, 'LineStyle', '--', 'LineWidth', 2.5, 'Color', dtf_color);

ylim([0, max_dtf_val]);
yticks(yticks_dtf);
ylabel('Spec. SD', 'FontSize', 14, 'FontWeight', 'bold', 'Color', dtf_color);
set(ax_psg_avg, 'YColor', dtf_color); 

xlim([700, 18000]);
xticks([1000, 2000, 4000, 8000, 16000]);
xticklabels({'1k', '2k', '4k', '8k', '16k'});
xlabel('Center Frequency (Hz)', 'FontSize', 16, 'FontWeight', 'bold');
grid on; set(ax_psg_avg, 'GridAlpha', 0.15); box on;
set(ax_psg_avg, 'Layer', 'top');

str_anno_avg = sprintf('$\\sigma_{\\mathrm{prior}}$: %.1f$\\pm$%.1f$^\\circ$, $\\sigma_{\\mathrm{motion}}$: %.1f$\\pm$%.1f$^\\circ$', ...
    mean_params(1), std_params(1), mean_params(2), std_params(2));
text(0.015, 0.88, str_anno_avg, 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
     'FontSize', 14, 'Interpreter', 'latex', 'BackgroundColor', [1 1 1 0.85], 'EdgeColor', [0.3 0.3 0.3], 'Margin', 3);

% ===================== 3. Export the figure to PDF =====================
save_pdf = 'Figure2.pdf';
fprintf('>>> Exporting Figure 2 PDF...\n');
try
    exportgraphics(fig, save_pdf, 'ContentType', 'vector', 'BackgroundColor', 'w');
    fprintf('>>> Figure exported successfully.\n>>> File path: %s\n', save_pdf);
catch
    warning('exportgraphics failed; using print to create PDF.');
    pos = fig.Position;
    set(fig, 'PaperUnits', 'points');
    set(fig, 'PaperPosition', [0 0 pos(3) pos(4)]);
    set(fig, 'PaperSize', [pos(3) pos(4)]);
    print(fig, save_pdf, '-dpdf', '-painters');
end