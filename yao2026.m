function m = yao2026(summary, Tmpl, Target)
% YAO2026  ECW model simulation for target localization predictions
%   Usage: [m] = yao2026(summary, Tmpl, Target)
%
%   Input parameters:
%     summary   : struct with ECW model parameters
%         alphas : cue weights vector for ITD, ILD and spectral features
%         prior  : elevation prior width (degrees)
%         motion : sensorimotor noise parameter (degrees)
%     Tmpl      : template struct from feature extraction
%         mu         : mean values for z-score normalization
%         std        : standard deviations for z-score normalization
%         features   : template feature matrix (N_templates x D)
%         grid_el    : template elevation values (N_templates x 1)
%         coords_cart: template directions in Cartesian coordinates (N_templates x 3)
%     Target    : target stimulus struct
%         features   : raw target feature vectors (N_trials x D)
%         target_cart: target directions in Cartesian coordinates (N_trials x 3)
%
%   Output parameters:
%     m  : N x 8 matrix containing target and simulated response
%         coordinates in spherical and horizontal-polar formats.
%         Columns are [target_sph_az, target_sph_el,
%                      sim_sph_az, sim_sph_el,
%                      target_lat, target_pol,
%                      sim_lat, sim_pol]
%
%   yao2026(...) computes ECW model predictions by applying weighted
%   Gaussian likelihoods over binaural and spectral cues, combining an
%   elevation prior with von Mises-Fisher motor noise sampling.
%   The output can be analyzed with barumerli2023_metrics or other
%   localization-error evaluation routines.
%
%   See also: barumerli2023, barumerli2023_featureextraction, barumerli2023_metrics
%

    alphas = reshape(summary.alphas, 1, length(summary.alphas));
    s_prior = summary.prior;
    s_motion = summary.motion;

    % --- 1. z-score normalization ---
    % Normalize target feature vectors using template mean and std so that
    % both target and template cues are in the same standardized space.
    X_raw = Target.features;
    X = bsxfun(@rdivide, bsxfun(@minus, X_raw, Tmpl.mu), Tmpl.std);
    Mu = Tmpl.features;

    Grid_El = Tmpl.grid_el;
    N = size(X, 1);
    M = size(Mu, 1);

    % --- 2. weighted likelihood computation ---
    % The first two weights apply to binaural cues (ITD, ILD) and the remaining
    % weights apply to spectral bands. Spectral weights are shared across left and
    % right ear representations, giving a total of D = 56 feature dimensions.
    log_det_term = 0.5 * sum(log(alphas + eps));
    sqrt_alphas = sqrt([alphas alphas(3:end)]);
    X_w = bsxfun(@times, X, sqrt_alphas);
    Mu_w = bsxfun(@times, Mu, sqrt_alphas);

    X_sq = sum(X_w.^2, 2);
    Mu_sq = sum(Mu_w.^2, 2)';
    w_dist_matrix = bsxfun(@plus, X_sq, Mu_sq) - 2 * (X_w * Mu_w');
    w_dist_matrix = max(w_dist_matrix, 0);

    % diagonal Gaussian log-likelihood with precision weights
    LogLikelihood = -0.5 * w_dist_matrix + log_det_term;
    Prob_L = exp(LogLikelihood - max(LogLikelihood, [], 2));

    % apply elevation prior and normalize to obtain valid posterior PMF
    Prior_dist = reshape(exp(-0.5 * (Grid_El / s_prior).^2), 1, M);
    Prob_Post = Prob_L .* (Prior_dist / (sum(Prior_dist) + eps));
    Prob_Post = Prob_Post ./ (sum(Prob_Post, 2) + eps);

    % --- 3. sensorimotor mapping using von Mises-Fisher motor noise ---
    % The motion parameter controls response dispersion in angular space.
    kappa = 1 / (deg2rad(s_motion)^2 + eps);
    Temp_Cart_norm = bsxfun(@rdivide, Tmpl.coords_cart, sqrt(sum(Tmpl.coords_cart.^2, 2)) + eps);

    sim_resp = zeros(N, 3);
    for n = 1:N
        cdf = cumsum(Prob_Post(n, :));
        max_idx = find(cdf >= rand(), 1, 'first');
        if isempty(max_idx), max_idx = M; end

        % sample a simulated response direction from a vMF distribution around
        % the chosen template direction with concentration kappa.
        sim_resp(n, :) = randvmf(kappa, Temp_Cart_norm(max_idx, :));
    end

    % --- 4. convert to output coordinate formats ---
    targets_cart = Target.target_cart;
    obj_tgt = barumerli2023_coordinates(targets_cart, 'cartesian');
    t_sph = obj_tgt.return_positions('spherical');
    t_lat = obj_tgt.return_positions('horizontal-polar');

    obj_sim = barumerli2023_coordinates(sim_resp, 'cartesian');
    sim_sph = obj_sim.return_positions('spherical');
    sim_lat = obj_sim.return_positions('horizontal-polar');

    m = [t_sph(:,1:2), sim_sph(:,1:2), t_lat(:,1:2), sim_lat(:,1:2)];
end