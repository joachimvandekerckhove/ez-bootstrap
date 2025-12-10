/*
 * EZ Circular Diffusion Model (EZ-CDM) - Common implementation
 * 
 * Shared functions for EZ-CDM Bootstrap implementations.
 */

#include "ezcdm_common.h"

/* Helper function to compute R = I1(kappa)/I0(kappa) using Taylor expansions for small kappa */
static double compute_R_from_kappa_taylor(double kappa) {
    if (kappa < 1e-6) {
        /* For very small kappa: Use Taylor expansion
         * I0(kappa) ≈ 1 + kappa^2/4 + kappa^4/64
         * I1(kappa) ≈ kappa/2 + kappa^3/16
         * R = I1/I0 ≈ (kappa/2 + kappa^3/16) / (1 + kappa^2/4 + kappa^4/64)
         * For very small kappa, R ≈ kappa/2
         */
        if (kappa < 1e-10) {
            return kappa / 2.0;
        }
        double k2 = kappa * kappa;
        double k4 = k2 * k2;
        double I0_approx = 1.0 + k2 / 4.0 + k4 / 64.0;
        double I1_approx = kappa / 2.0 + kappa * k2 / 16.0;
        return I1_approx / I0_approx;
    }
    /* For larger kappa, use GSL Bessel functions */
    double I0 = gsl_sf_bessel_I0(kappa);
    double I1 = gsl_sf_bessel_I1(kappa);
    if (I0 <= 0.0) {
        return 0.0;
    }
    return I1 / I0;
}

/* Helper function to solve for kappa from R = I1(kappa)/I0(kappa)
 * Uses Fisher's piecewise approximation (Fisher, 1993, p. 88)
 * with Taylor expansion for very small R
 */
double solve_kappa_from_R(double R) {
    if (R <= 0.0 || R >= 1.0) {
        return 0.0;  /* Invalid R */
    }
    
    /* Newton-Raphson method from Qarehdaghi & Rad (2024)
     * Solving R = I₁(κ)/I₀(κ) for κ
     * 
     * Step 1: Initial approximation from Banerjee et al. (2005)
     * κ₀ = R(2 - R²)/(1 - R²)
     */
    double R_sq = R * R;
    double kappa_0 = R * (2.0 - R_sq) / (1.0 - R_sq);
    
    /* Handle edge case where R is very close to 1 */
    if (kappa_0 <= 0.0 || kappa_0 != kappa_0 || kappa_0 > 1e10) {  /* Check for NaN/inf */
        /* Fallback for R very close to 1 */
        if (R > 0.99) {
            return 100.0;  /* Large kappa for R near 1 */
        }
        kappa_0 = 2.0 * R;  /* Simple approximation for very small R */
    }
    
    /* Step 2: One Newton-Raphson iteration
     * κ₁ = κ₀ - [I₁(κ₀)/I₀(κ₀) - R] / [1 - I₁²(κ₀)/I₀²(κ₀) - I₁(κ₀)/(κ₀·I₀(κ₀))]
     */
    double I0_k0 = gsl_sf_bessel_I0(kappa_0);
    double I1_k0 = gsl_sf_bessel_I1(kappa_0);
    
    if (I0_k0 <= 0.0 || I0_k0 != I0_k0 || I1_k0 != I1_k0) {  /* Check for NaN/inf */
        return kappa_0;  /* Return initial approximation if Bessel functions fail */
    }
    
    double R_k0 = I1_k0 / I0_k0;  /* I₁(κ₀)/I₀(κ₀) */
    double R_k0_sq = R_k0 * R_k0;
    
    /* Denominator: 1 - [I₁(κ₀)/I₀(κ₀)]² - I₁(κ₀)/(κ₀·I₀(κ₀)) */
    double denominator = 1.0 - R_k0_sq - I1_k0 / (kappa_0 * I0_k0);
    
    /* Check if denominator is valid */
    if (fabs(denominator) < 1e-10 || denominator != denominator) {  /* Check for NaN */
        return kappa_0;  /* Return initial approximation if denominator is too small */
    }
    
    /* Numerator: I₁(κ₀)/I₀(κ₀) - R */
    double numerator = R_k0 - R;
    
    /* Newton-Raphson step */
    double kappa_1 = kappa_0 - numerator / denominator;
    
    /* Validate result */
    if (kappa_1 <= 0.0 || kappa_1 != kappa_1 || kappa_1 > 1e10) {  /* Check for NaN/inf */
        return kappa_0;  /* Return initial approximation if Newton-Raphson fails */
    }
    
    return kappa_1;
}

/* EZ-CDM forward equations: transform parameters to summary statistics */
CDMSummaryStats ezcdm_forward(CDMParameters params) {
    CDMSummaryStats moments = {0.0, 0.0, 0.0, 0.0, 0};
    
    if (params.boundary_radius <= 0.0 || params.drift_mag <= 0.0) {
        return moments;
    }
    
    /* MCA = drift_angle (direct) */
    moments.mca = params.drift_angle;
    
    /* Compute kappa = boundary_radius * drift_mag */
    double kappa = params.boundary_radius * params.drift_mag;
    
    /* Compute R = I1(kappa) / I0(kappa) using Taylor expansion for small kappa */
    double R = compute_R_from_kappa_taylor(kappa);
    
    if (R <= 0.0 || R >= 1.0) {
        return moments;  /* Invalid R */
    }
    
    /* VCA = 1 - R */
    moments.vca = 1.0 - R;
    
    /* MRT = ndt + (boundary_radius/drift_mag) * R */
    moments.mrt = params.ndt + (params.boundary_radius / params.drift_mag) * R;
    
    /* VRT = (boundary_radius^2/drift_mag^2) * R^2 + (2*boundary_radius/drift_mag^3) * R - boundary_radius^2 */
    double r_over_mu = params.boundary_radius / params.drift_mag;
    double r_over_mu_sq = r_over_mu * r_over_mu;
    double r_over_mu_cubed = r_over_mu_sq / params.drift_mag;
    moments.vrt = r_over_mu_sq * R * R + 2.0 * r_over_mu_cubed * R - params.boundary_radius * params.boundary_radius;
    
    /* Check if VRT is valid (must be positive for a variance) */
    if (moments.vrt <= 0.0) {
        /* Invalid VRT - return zero moments to indicate failure */
        CDMSummaryStats invalid = {0.0, 0.0, 0.0, 0.0, 0};
        return invalid;
    }
    
    return moments;
}

/* EZ-CDM inverse equations: transform summary statistics to parameters */
CDMParameters ezcdm_inverse(CDMSummaryStats stats) {
    CDMParameters params = {0.0, 0.0, 0.0, 0.0};
    
    double mca = stats.mca;
    double vca = stats.vca;
    double mrt = stats.mrt;
    double vrt = stats.vrt;
    int n = stats.n;
    
    /* Handle edge cases */
    if (vca < 0.0 || vca > 1.0 || vrt <= 0.0 || n <= 0) {
        return params;
    }
    
    /* Step 1: drift_angle = MCA (direct) */
    params.drift_angle = mca;
    
    /* Step 2: R = 1 - VCA */
    double R = 1.0 - vca;
    
    if (R <= 0.0 || R >= 1.0) {
        return params;  /* Invalid R */
    }
    
    /* Step 3: Solve R = I1(kappa)/I0(kappa) for kappa */
    double kappa = solve_kappa_from_R(R);
    
    if (kappa <= 0.0) {
        return params;  /* Failed to solve for kappa */
    }
    
    /* Step 4: Solve for drift_mag using PDF inverse equation
     * From forward: VRT = (a²/v²) R² + (2a/v³) R - a²/v²
     * where a = boundary_radius, v = drift_mag, κ = av
     * 
     * Substituting a = κ/v and rearranging:
     * VRT = (κ²/v⁴) R² + (2κ/v⁴) R - κ²/v⁴
     * VRT · v⁴ = κ² R² + 2κ R - κ²
     * v⁴ = (κ² R² + 2κ R - κ²) / VRT
     * v = [(κ² R² + 2κ R - κ²) / VRT]^(1/4)
     */
    if (vrt <= 0.0) {
        return params;
    }
    
    double kappa_sq = kappa * kappa;
    double R_sq = R * R;
    
    /* Compute v⁴ = (κ² R² + 2κ R - κ²) / VRT */
    double numerator = kappa_sq * R_sq + 2.0 * kappa * R - kappa_sq;
    
    if (numerator <= 0.0) {
        return params;  /* Invalid numerator */
    }
    
    double v_4 = numerator / vrt;
    
    if (v_4 <= 0.0) {
        return params;  /* Invalid v⁴ */
    }
    
    /* Compute v = (v⁴)^(1/4) = sqrt(sqrt(v⁴)) */
    params.drift_mag = sqrt(sqrt(v_4));
    
    /* Additional check: if drift_mag is too small or too large, it's likely invalid */
    if (params.drift_mag < 1e-4 || params.drift_mag > 50.0) {
        return params;  /* Invalid drift magnitude */
    }
    
    /* Step 5: boundary_radius = kappa / drift_mag */
    if (params.drift_mag <= 0.0) {
        return params;  /* Invalid drift magnitude */
    }
    params.boundary_radius = kappa / params.drift_mag;
    
    /* Additional check: boundary_radius should be positive and reasonable */
    if (params.boundary_radius <= 0.0 || params.boundary_radius > 50.0) {
        return params;  /* Invalid boundary radius */
    }
    
    /* Step 6: ndt = MRT - (boundary_radius/drift_mag) * R */
    /* Use Taylor expansion for R if kappa is small */
    double R_for_ndt = (kappa < 1e-3) ? compute_R_from_kappa_taylor(kappa) : R;
    double ndt_term = (params.boundary_radius / params.drift_mag) * R_for_ndt;
    
    /* Check if the subtraction would produce invalid result */
    if (ndt_term > mrt || ndt_term < 0.0) {
        return params;  /* Invalid computation */
    }
    
    params.ndt = mrt - ndt_term;
    
    /* Check: ndt should be non-negative and reasonable */
    if (params.ndt < 0.0 || params.ndt > 5.0) {
        return params;  /* Invalid NDT */
    }
    
    return params;
}

/* Sample observations from moments using GSL */
CDMSummaryStats sample_cdm_observations(CDMSummaryStats moments, int sample_size, const gsl_rng *r) {
    CDMSummaryStats obs;
    obs.n = sample_size;
    
    /* For MCA: Normal(θ, 1/(N×κ×A₁(κ))) where A₁(κ) = I₁(κ)/I₀(κ)
     * We need to compute kappa from the moments. Since we have VCA, we can get R = 1-VCA,
     * then solve for kappa. However, for sampling we need the true kappa.
     * For now, we'll approximate using the observed VCA to estimate kappa.
     */
    double R_obs = 1.0 - moments.vca;
    double kappa_est = solve_kappa_from_R(R_obs);
    double A1 = 0.0;
    if (kappa_est > 0.0) {
        A1 = compute_R_from_kappa_taylor(kappa_est);
    }
    
    /* Sample MCA from normal */
    double mca_var = 1.0 / (sample_size * kappa_est * A1);
    if (mca_var <= 0.0 || kappa_est <= 0.0 || A1 <= 0.0) {
        mca_var = 0.01;  /* Fallback for edge cases */
    }
    obs.mca = gsl_ran_gaussian(r, sqrt(mca_var)) + moments.mca;
    /* Wrap to [-π, π] */
    while (obs.mca > M_PI) obs.mca -= 2.0 * M_PI;
    while (obs.mca < -M_PI) obs.mca += 2.0 * M_PI;
    
    /* For VCA: Normal(1-A₁(κ), (1-A₁(κ)²)/N) */
    double vca_mean = 1.0 - A1;
    double vca_var = (1.0 - A1 * A1) / sample_size;
    if (vca_var <= 0.0) {
        vca_var = 0.001;  /* Fallback */
    }
    obs.vca = gsl_ran_gaussian(r, sqrt(vca_var)) + vca_mean;
    /* Clamp to [0, 1] */
    if (obs.vca < 0.0) obs.vca = 0.0;
    if (obs.vca > 1.0) obs.vca = 1.0;
    
    /* For MRT: Normal(E[T], Var(T)/N) via CLT */
    double mrt_std = sqrt(moments.vrt / sample_size);
    obs.mrt = gsl_ran_gaussian(r, mrt_std) + moments.mrt;
    if (obs.mrt < 0.0) obs.mrt = 0.0;  /* RT must be positive */
    
    /* For VRT: Normal(Var(T), (μ₄-Var(T)²)/N) via CLT
     * We approximate μ₄ ≈ 3*Var(T)² for normal distribution
     */
    double vrt_mean = moments.vrt;
    double vrt_var = (2.0 * moments.vrt * moments.vrt) / sample_size;  /* Approximate */
    if (vrt_var <= 0.0) {
        vrt_var = 0.001;  /* Fallback */
    }
    obs.vrt = gsl_ran_gaussian(r, sqrt(vrt_var)) + vrt_mean;
    if (obs.vrt <= 0.0) obs.vrt = 0.001;  /* Variance must be positive */
    
    return obs;
}

/* Sample bootstrap summary statistics from their distributions using GSL */
CDMSummaryStats sample_cdm_bootstrap_stats(CDMSummaryStats observed, const gsl_rng *r) {
    CDMSummaryStats boot;
    
    /* Estimate kappa from observed VCA */
    double R_obs = 1.0 - observed.vca;
    double kappa_est = solve_kappa_from_R(R_obs);
    double A1 = 0.0;
    if (kappa_est > 0.0) {
        double I0 = gsl_sf_bessel_I0(kappa_est);
        double I1 = gsl_sf_bessel_I1(kappa_est);
        if (I0 > 0.0) {
            A1 = I1 / I0;
        }
    }
    
    /* Sample MCA from normal */
    double mca_var = 1.0 / (observed.n * kappa_est * A1);
    if (mca_var <= 0.0 || kappa_est <= 0.0 || A1 <= 0.0) {
        mca_var = 0.01;  /* Fallback */
    }
    boot.mca = gsl_ran_gaussian(r, sqrt(mca_var)) + observed.mca;
    /* Wrap to [-π, π] */
    while (boot.mca > M_PI) boot.mca -= 2.0 * M_PI;
    while (boot.mca < -M_PI) boot.mca += 2.0 * M_PI;
    
    /* Sample VCA from normal */
    double vca_mean = 1.0 - A1;
    double vca_var = (1.0 - A1 * A1) / observed.n;
    if (vca_var <= 0.0) {
        vca_var = 0.001;  /* Fallback */
    }
    boot.vca = gsl_ran_gaussian(r, sqrt(vca_var)) + vca_mean;
    /* Clamp to [0, 1] */
    if (boot.vca < 0.0) boot.vca = 0.0;
    if (boot.vca > 1.0) boot.vca = 1.0;
    
    /* Sample MRT from normal */
    double mrt_std = sqrt(observed.vrt / observed.n);
    boot.mrt = gsl_ran_gaussian(r, mrt_std) + observed.mrt;
    if (boot.mrt < 0.0) boot.mrt = 0.0;
    
    /* Sample VRT from normal */
    double vrt_mean = observed.vrt;
    double vrt_var = (2.0 * observed.vrt * observed.vrt) / observed.n;
    if (vrt_var <= 0.0) {
        vrt_var = 0.001;  /* Fallback */
    }
    boot.vrt = gsl_ran_gaussian(r, sqrt(vrt_var)) + vrt_mean;
    if (boot.vrt <= 0.0) boot.vrt = 0.001;
    
    boot.n = observed.n;
    
    return boot;
}

/* Run bootstrap using GSL random number generator */
CDMBootstrapResults cdm_bootstrap(CDMSummaryStats observed, int n_bootstrap, const gsl_rng *r) {
    CDMBootstrapResults results;
    results.n_samples = n_bootstrap;
    results.drift_angle = malloc(n_bootstrap * sizeof(double));
    results.drift_mag = malloc(n_bootstrap * sizeof(double));
    results.boundary_radius = malloc(n_bootstrap * sizeof(double));
    results.ndt = malloc(n_bootstrap * sizeof(double));
    
    if (!results.drift_angle || !results.drift_mag || 
        !results.boundary_radius || !results.ndt) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < n_bootstrap; i++) {
        CDMSummaryStats boot_stats = sample_cdm_bootstrap_stats(observed, r);
        CDMParameters params = ezcdm_inverse(boot_stats);
        results.drift_angle[i] = params.drift_angle;
        results.drift_mag[i] = params.drift_mag;
        results.boundary_radius[i] = params.boundary_radius;
        results.ndt[i] = params.ndt;
    }
    
    return results;
}

/* Print summary statistics using GSL statistics functions */
void print_cdm_summary(CDMBootstrapResults results) {
    printf("Bootstrap Results (n=%d):\n", results.n_samples);
    
    /* Compute statistics using GSL */
    double drift_angle_mean = gsl_stats_mean(results.drift_angle, 1, results.n_samples);
    double drift_angle_sd = gsl_stats_sd_m(results.drift_angle, 1, results.n_samples, drift_angle_mean);
    double drift_mag_mean = gsl_stats_mean(results.drift_mag, 1, results.n_samples);
    double drift_mag_sd = gsl_stats_sd_m(results.drift_mag, 1, results.n_samples, drift_mag_mean);
    double boundary_radius_mean = gsl_stats_mean(results.boundary_radius, 1, results.n_samples);
    double boundary_radius_sd = gsl_stats_sd_m(results.boundary_radius, 1, results.n_samples, boundary_radius_mean);
    double ndt_mean = gsl_stats_mean(results.ndt, 1, results.n_samples);
    double ndt_sd = gsl_stats_sd_m(results.ndt, 1, results.n_samples, ndt_mean);
    
    /* Compute percentiles using GSL sort */
    double *drift_angle_sorted = malloc(results.n_samples * sizeof(double));
    double *drift_mag_sorted = malloc(results.n_samples * sizeof(double));
    double *boundary_radius_sorted = malloc(results.n_samples * sizeof(double));
    double *ndt_sorted = malloc(results.n_samples * sizeof(double));
    
    if (!drift_angle_sorted || !drift_mag_sorted || 
        !boundary_radius_sorted || !ndt_sorted) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    memcpy(drift_angle_sorted, results.drift_angle, results.n_samples * sizeof(double));
    memcpy(drift_mag_sorted, results.drift_mag, results.n_samples * sizeof(double));
    memcpy(boundary_radius_sorted, results.boundary_radius, results.n_samples * sizeof(double));
    memcpy(ndt_sorted, results.ndt, results.n_samples * sizeof(double));
    
    gsl_sort(drift_angle_sorted, 1, results.n_samples);
    gsl_sort(drift_mag_sorted, 1, results.n_samples);
    gsl_sort(boundary_radius_sorted, 1, results.n_samples);
    gsl_sort(ndt_sorted, 1, results.n_samples);
    
    int idx_025 = (int)(0.025 * (results.n_samples - 1));
    int idx_975 = (int)(0.975 * (results.n_samples - 1));
    
    printf("\nDrift angle (theta):\n");
    printf("  Mean: %.4f\n", drift_angle_mean);
    printf("  Std:  %.4f\n", drift_angle_sd);
    printf("  2.5%%:  %.4f\n", drift_angle_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", drift_angle_sorted[idx_975]);
    
    printf("\nDrift magnitude (mu):\n");
    printf("  Mean: %.4f\n", drift_mag_mean);
    printf("  Std:  %.4f\n", drift_mag_sd);
    printf("  2.5%%:  %.4f\n", drift_mag_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", drift_mag_sorted[idx_975]);
    
    printf("\nBoundary radius (r):\n");
    printf("  Mean: %.4f\n", boundary_radius_mean);
    printf("  Std:  %.4f\n", boundary_radius_sd);
    printf("  2.5%%:  %.4f\n", boundary_radius_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", boundary_radius_sorted[idx_975]);
    
    printf("\nNon-decision time (tau):\n");
    printf("  Mean: %.4f\n", ndt_mean);
    printf("  Std:  %.4f\n", ndt_sd);
    printf("  2.5%%:  %.4f\n", ndt_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", ndt_sorted[idx_975]);
    
    free(drift_angle_sorted);
    free(drift_mag_sorted);
    free(boundary_radius_sorted);
    free(ndt_sorted);
}

/* Free bootstrap results */
void free_cdm_bootstrap_results(CDMBootstrapResults results) {
    free(results.drift_angle);
    free(results.drift_mag);
    free(results.boundary_radius);
    free(results.ndt);
}

/* Compute credible intervals from bootstrap results */
static void compute_cdm_intervals(CDMBootstrapResults results, 
                                  double *drift_angle_lower, double *drift_angle_upper,
                                  double *drift_mag_lower, double *drift_mag_upper,
                                  double *boundary_radius_lower, double *boundary_radius_upper,
                                  double *ndt_lower, double *ndt_upper) {
    /* Create sorted copies */
    double *drift_angle_sorted = malloc(results.n_samples * sizeof(double));
    double *drift_mag_sorted = malloc(results.n_samples * sizeof(double));
    double *boundary_radius_sorted = malloc(results.n_samples * sizeof(double));
    double *ndt_sorted = malloc(results.n_samples * sizeof(double));
    
    if (!drift_angle_sorted || !drift_mag_sorted || 
        !boundary_radius_sorted || !ndt_sorted) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    memcpy(drift_angle_sorted, results.drift_angle, results.n_samples * sizeof(double));
    memcpy(drift_mag_sorted, results.drift_mag, results.n_samples * sizeof(double));
    memcpy(boundary_radius_sorted, results.boundary_radius, results.n_samples * sizeof(double));
    memcpy(ndt_sorted, results.ndt, results.n_samples * sizeof(double));
    
    gsl_sort(drift_angle_sorted, 1, results.n_samples);
    gsl_sort(drift_mag_sorted, 1, results.n_samples);
    gsl_sort(boundary_radius_sorted, 1, results.n_samples);
    gsl_sort(ndt_sorted, 1, results.n_samples);
    
    int idx_025 = (int)(0.025 * (results.n_samples - 1));
    int idx_975 = (int)(0.975 * (results.n_samples - 1));
    
    *drift_angle_lower = drift_angle_sorted[idx_025];
    *drift_angle_upper = drift_angle_sorted[idx_975];
    *drift_mag_lower = drift_mag_sorted[idx_025];
    *drift_mag_upper = drift_mag_sorted[idx_975];
    *boundary_radius_lower = boundary_radius_sorted[idx_025];
    *boundary_radius_upper = boundary_radius_sorted[idx_975];
    *ndt_lower = ndt_sorted[idx_025];
    *ndt_upper = ndt_sorted[idx_975];
    
    free(drift_angle_sorted);
    free(drift_mag_sorted);
    free(boundary_radius_sorted);
    free(ndt_sorted);
}

/* Check if true parameters are within estimated intervals */
int check_cdm_coverage(CDMParameters true_params, CDMBootstrapResults results) {
    double drift_angle_lower, drift_angle_upper;
    double drift_mag_lower, drift_mag_upper;
    double boundary_radius_lower, boundary_radius_upper;
    double ndt_lower, ndt_upper;
    
    compute_cdm_intervals(results, &drift_angle_lower, &drift_angle_upper,
                         &drift_mag_lower, &drift_mag_upper,
                         &boundary_radius_lower, &boundary_radius_upper,
                         &ndt_lower, &ndt_upper);
    
    int drift_angle_covered = (drift_angle_lower <= true_params.drift_angle && 
                                true_params.drift_angle <= drift_angle_upper);
    int drift_mag_covered = (drift_mag_lower <= true_params.drift_mag && 
                            true_params.drift_mag <= drift_mag_upper);
    int boundary_radius_covered = (boundary_radius_lower <= true_params.boundary_radius && 
                                   true_params.boundary_radius <= boundary_radius_upper);
    int ndt_covered = (ndt_lower <= true_params.ndt && 
                      true_params.ndt <= ndt_upper);
    
    return (drift_angle_covered << 3) | (drift_mag_covered << 2) | 
           (boundary_radius_covered << 1) | ndt_covered;
}
