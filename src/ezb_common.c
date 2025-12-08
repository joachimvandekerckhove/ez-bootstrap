/*
 * EZ Bootstrap (EZB) - Common implementation
 * 
 * Shared functions for EZ Bootstrap implementations.
 */

#include "ezb_common.h"

/* EZ inverse equations: transform summary statistics to parameters */
Parameters ez_inverse(SummaryStats stats) {
    Parameters params = {0.0, 0.0, 0.0};
    double p = stats.accuracy;
    double m = stats.mean_rt;
    double s2 = stats.var_rt;
    int n = stats.n;
    
    /* Handle edge cases */
    if (p <= 0.0 || p >= 1.0 || s2 <= 0.0 || n <= 0) {
        return params;
    }
    
    /* Compute logit */
    double logit = log(p / (1.0 - p));
    double y = logit;
    
    /* Compute numerator for drift */
    double numerator = y * (p * p * y - p * y + p - 0.5);
    
    if (numerator <= 0.0 || s2 <= 0.0) {
        return params;
    }
    
    /* Drift rate */
    int sign = (p > 0.5) ? 1 : -1;
    double drift = sign * pow(numerator / s2, 0.25);
    
    if (fabs(drift) < 1e-9) {
        params.boundary = 1.0;
        params.drift = 0.0;
        params.ndt = m;
        return params;
    }
    
    /* Boundary separation */
    double boundary = fabs(y / drift);
    
    /* Non-decision time */
    double exp_term = exp(-drift * boundary);
    double ndt = m - (boundary / (2.0 * drift)) * ((1.0 - exp_term) / (1.0 + exp_term));
    
    params.boundary = boundary;
    params.drift = drift;
    params.ndt = ndt;
    
    return params;
}

/* Forward equations: transform parameters to moments */
SummaryStats ez_forward(Parameters params) {
    SummaryStats moments = {0.0, 0.0, 0.0, 0};
    
    if (params.boundary <= 0.0) {
        return moments;
    }
    
    if (fabs(params.drift) < 1e-9) {
        /* Special case: drift = 0 */
        moments.accuracy = 0.5;
        moments.mean_rt = params.ndt + params.boundary;
        moments.var_rt = params.boundary * params.boundary;
        return moments;
    }
    
    double y = exp(-params.boundary * params.drift);
    moments.accuracy = 1.0 / (y + 1.0);
    moments.mean_rt = params.ndt + (params.boundary / (2.0 * params.drift)) * ((1.0 - y) / (1.0 + y));
    
    double numerator = 1.0 - 2.0 * params.boundary * params.drift * y - y * y;
    double denominator = (y + 1.0) * (y + 1.0);
    moments.var_rt = (params.boundary / (2.0 * params.drift * params.drift * params.drift)) * (numerator / denominator);
    
    return moments;
}

/* Sample observations from moments using GSL */
SummaryStats sample_observations(SummaryStats moments, int sample_size, const gsl_rng *r) {
    SummaryStats obs;
    obs.n = sample_size;
    
    /* Sample accuracy from binomial */
    unsigned int n_correct = gsl_ran_binomial(r, moments.accuracy, (unsigned int)sample_size);
    obs.accuracy = (double)n_correct / sample_size;
    
    /* Sample mean RT from normal */
    double mean_std = sqrt(moments.var_rt / sample_size);
    obs.mean_rt = gsl_ran_gaussian(r, mean_std) + moments.mean_rt;
    
    /* Sample variance from gamma */
    double shape = (sample_size - 1.0) / 2.0;
    double scale = 2.0 * moments.var_rt / (sample_size - 1.0);
    obs.var_rt = gsl_ran_gamma(r, shape, scale);
    
    return obs;
}

/* Sample bootstrap summary statistics from their distributions using GSL */
SummaryStats sample_bootstrap_stats(SummaryStats observed, const gsl_rng *r) {
    SummaryStats boot;
    
    /* Sample accuracy from binomial using GSL */
    unsigned int n_correct = gsl_ran_binomial(r, observed.accuracy, (unsigned int)observed.n);
    boot.accuracy = (double)n_correct / observed.n;
    
    /* Sample mean RT from normal using GSL */
    double mean_std = sqrt(observed.var_rt / observed.n);
    boot.mean_rt = gsl_ran_gaussian(r, mean_std) + observed.mean_rt;
    
    /* Sample variance from gamma using GSL (shape-scale parameterization) */
    double shape = (observed.n - 1.0) / 2.0;
    double scale = 2.0 * observed.var_rt / (observed.n - 1.0);
    boot.var_rt = gsl_ran_gamma(r, shape, scale);
    
    boot.n = observed.n;
    
    return boot;
}

/* Run bootstrap using GSL random number generator */
BootstrapResults bootstrap(SummaryStats observed, int n_bootstrap, const gsl_rng *r) {
    BootstrapResults results;
    results.n_samples = n_bootstrap;
    results.boundary = malloc(n_bootstrap * sizeof(double));
    results.drift = malloc(n_bootstrap * sizeof(double));
    results.ndt = malloc(n_bootstrap * sizeof(double));
    
    if (!results.boundary || !results.drift || !results.ndt) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    for (int i = 0; i < n_bootstrap; i++) {
        SummaryStats boot_stats = sample_bootstrap_stats(observed, r);
        Parameters params = ez_inverse(boot_stats);
        results.boundary[i] = params.boundary;
        results.drift[i] = params.drift;
        results.ndt[i] = params.ndt;
    }
    
    return results;
}

/* Print summary statistics using GSL statistics functions */
void print_summary(BootstrapResults results) {
    printf("Bootstrap Results (n=%d):\n", results.n_samples);
    
    /* Compute statistics using GSL */
    double boundary_mean = gsl_stats_mean(results.boundary, 1, results.n_samples);
    double boundary_sd = gsl_stats_sd_m(results.boundary, 1, results.n_samples, boundary_mean);
    double drift_mean = gsl_stats_mean(results.drift, 1, results.n_samples);
    double drift_sd = gsl_stats_sd_m(results.drift, 1, results.n_samples, drift_mean);
    double ndt_mean = gsl_stats_mean(results.ndt, 1, results.n_samples);
    double ndt_sd = gsl_stats_sd_m(results.ndt, 1, results.n_samples, ndt_mean);
    
    /* Compute percentiles using GSL sort */
    double *boundary_sorted = malloc(results.n_samples * sizeof(double));
    double *drift_sorted = malloc(results.n_samples * sizeof(double));
    double *ndt_sorted = malloc(results.n_samples * sizeof(double));
    
    if (!boundary_sorted || !drift_sorted || !ndt_sorted) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    memcpy(boundary_sorted, results.boundary, results.n_samples * sizeof(double));
    memcpy(drift_sorted, results.drift, results.n_samples * sizeof(double));
    memcpy(ndt_sorted, results.ndt, results.n_samples * sizeof(double));
    
    gsl_sort(boundary_sorted, 1, results.n_samples);
    gsl_sort(drift_sorted, 1, results.n_samples);
    gsl_sort(ndt_sorted, 1, results.n_samples);
    
    int idx_025 = (int)(0.025 * (results.n_samples - 1));
    int idx_975 = (int)(0.975 * (results.n_samples - 1));
    
    printf("\nBoundary separation (alpha):\n");
    printf("  Mean: %.4f\n", boundary_mean);
    printf("  Std:  %.4f\n", boundary_sd);
    printf("  2.5%%:  %.4f\n", boundary_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", boundary_sorted[idx_975]);
    
    printf("\nDrift rate (delta):\n");
    printf("  Mean: %.4f\n", drift_mean);
    printf("  Std:  %.4f\n", drift_sd);
    printf("  2.5%%:  %.4f\n", drift_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", drift_sorted[idx_975]);
    
    printf("\nNon-decision time (tau):\n");
    printf("  Mean: %.4f\n", ndt_mean);
    printf("  Std:  %.4f\n", ndt_sd);
    printf("  2.5%%:  %.4f\n", ndt_sorted[idx_025]);
    printf("  97.5%%: %.4f\n", ndt_sorted[idx_975]);
    
    free(boundary_sorted);
    free(drift_sorted);
    free(ndt_sorted);
}

/* Free bootstrap results */
void free_bootstrap_results(BootstrapResults results) {
    free(results.boundary);
    free(results.drift);
    free(results.ndt);
}

/* Compute credible intervals from bootstrap results */
static void compute_intervals(BootstrapResults results, 
                              double *boundary_lower, double *boundary_upper,
                              double *drift_lower, double *drift_upper,
                              double *ndt_lower, double *ndt_upper) {
    /* Create sorted copies */
    double *boundary_sorted = malloc(results.n_samples * sizeof(double));
    double *drift_sorted = malloc(results.n_samples * sizeof(double));
    double *ndt_sorted = malloc(results.n_samples * sizeof(double));
    
    if (!boundary_sorted || !drift_sorted || !ndt_sorted) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    memcpy(boundary_sorted, results.boundary, results.n_samples * sizeof(double));
    memcpy(drift_sorted, results.drift, results.n_samples * sizeof(double));
    memcpy(ndt_sorted, results.ndt, results.n_samples * sizeof(double));
    
    gsl_sort(boundary_sorted, 1, results.n_samples);
    gsl_sort(drift_sorted, 1, results.n_samples);
    gsl_sort(ndt_sorted, 1, results.n_samples);
    
    int idx_025 = (int)(0.025 * (results.n_samples - 1));
    int idx_975 = (int)(0.975 * (results.n_samples - 1));
    
    *boundary_lower = boundary_sorted[idx_025];
    *boundary_upper = boundary_sorted[idx_975];
    *drift_lower = drift_sorted[idx_025];
    *drift_upper = drift_sorted[idx_975];
    *ndt_lower = ndt_sorted[idx_025];
    *ndt_upper = ndt_sorted[idx_975];
    
    free(boundary_sorted);
    free(drift_sorted);
    free(ndt_sorted);
}

/* Check if true parameters are within estimated intervals */
int check_coverage(Parameters true_params, BootstrapResults results) {
    double boundary_lower, boundary_upper;
    double drift_lower, drift_upper;
    double ndt_lower, ndt_upper;
    
    compute_intervals(results, &boundary_lower, &boundary_upper,
                     &drift_lower, &drift_upper,
                     &ndt_lower, &ndt_upper);
    
    int boundary_covered = (boundary_lower <= true_params.boundary && 
                            true_params.boundary <= boundary_upper);
    int drift_covered = (drift_lower <= true_params.drift && 
                        true_params.drift <= drift_upper);
    int ndt_covered = (ndt_lower <= true_params.ndt && 
                      true_params.ndt <= ndt_upper);
    
    return (boundary_covered << 2) | (drift_covered << 1) | ndt_covered;
}

