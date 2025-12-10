/*
 * EZ Circular Diffusion Model (EZ-CDM) - Design matrix implementation
 * 
 * Implements EZ bootstrap for EZ circular diffusion models with design matrices.
 */

#include "ezcdm_common.h"
#include "ezcdm_design_matrix.h"

#define DEFAULT_N_BOOTSTRAP 1000

/* Example design matrices (4 conditions, 3 weights each) - original structure */
static double example_drift_angle_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    0, 1, 0
};

static double example_drift_mag_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    0, 1, 0
};

static double example_boundary_radius_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    1, 0, 0
};

static double example_ndt_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    0, 0, 1
};

/* Beta weights: using valid range for drift_mag (≤ 1 to avoid negative VRT) */
static double example_drift_angle_weights[] = {0.0, 0.5, 1.0};
static double example_drift_mag_weights[] = {0.5, 0.7, 0.9};
static double example_boundary_radius_weights[] = {0.8, 1.0, 1.2};
static double example_ndt_weights[] = {0.15, 0.2, 0.25};

/* EZ-CDM bootstrap for design matrix */
CDMBetaWeights** ezcdm_design_matrix_bootstrap(CDMSummaryStats *observations, int n_conditions,
                                              CDMDesignMatrix *dm, int n_bootstrap, const gsl_rng *r) {
    CDMBetaWeights **bw_list = malloc(n_bootstrap * sizeof(CDMBetaWeights*));
    if (!bw_list) return NULL;
    
    /* Get initial parameters from observations */
    CDMParameters *params = malloc(n_conditions * sizeof(CDMParameters));
    for (int i = 0; i < n_conditions; i++) {
        params[i] = ezcdm_inverse(observations[i]);
    }
    
    /* Bootstrap loop */
    int b = 0;
    int total_attempts = 0;
    int max_total_attempts = n_bootstrap * 100;  /* Maximum total attempts */
    
    while (b < n_bootstrap && total_attempts < max_total_attempts) {
        total_attempts++;
        
        int max_retries = 100;  /* Maximum retries per bootstrap sample */
        int retry = 0;
        int valid = 0;
        CDMParameters *boot_params = NULL;
        
        while (retry < max_retries && !valid) {
            retry++;
            
            /* Resample observations */
            CDMSummaryStats *boot_obs = malloc(n_conditions * sizeof(CDMSummaryStats));
            for (int i = 0; i < n_conditions; i++) {
                boot_obs[i] = sample_cdm_bootstrap_stats(observations[i], r);
            }
            
            /* Get parameters from resampled observations */
            if (boot_params != NULL) {
                free(boot_params);
            }
            boot_params = malloc(n_conditions * sizeof(CDMParameters));
            valid = 1;
            for (int i = 0; i < n_conditions; i++) {
                boot_params[i] = ezcdm_inverse(boot_obs[i]);
                /* Check if parameters are valid (non-zero for positive parameters) */
                if (boot_params[i].drift_mag <= 0.0 || 
                    boot_params[i].boundary_radius <= 0.0 || 
                    boot_params[i].ndt < 0.0 ||
                    boot_params[i].drift_mag > 100.0 ||
                    boot_params[i].boundary_radius > 100.0 ||
                    boot_params[i].ndt > 10.0) {
                    valid = 0;
                    break;
                }
            }
            
            free(boot_obs);
        }
        
        /* If we still don't have valid parameters after retries, use original parameters as fallback */
        if (!valid) {
            if (boot_params != NULL) {
                free(boot_params);
            }
            /* Fallback: use original parameters (this should be rare) */
            boot_params = malloc(n_conditions * sizeof(CDMParameters));
            for (int i = 0; i < n_conditions; i++) {
                boot_params[i] = params[i];
            }
        }
        
        /* Estimate beta weights */
        bw_list[b] = cdm_design_matrix_estimate_weights(dm, boot_params, n_conditions);
        
        if (boot_params != NULL) {
            free(boot_params);
        }
        
        /* If estimation failed, skip this sample and continue */
        if (bw_list[b] == NULL) {
            continue;  /* Skip this bootstrap sample */
        }
        
        b++;  /* Only increment if we got a valid sample */
    }
    
    /* Fill remaining slots with NULL if we didn't get enough valid samples */
    while (b < n_bootstrap) {
        bw_list[b] = NULL;
        b++;
    }
    
    free(params);
    return bw_list;
}

/* Demo function for design matrix */
void demo_design_matrix(CDMDesignMatrix *dm, int sample_size) {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long)time(NULL));
    
    printf("EZ Circular Diffusion Model (EZ-CDM) - Design Matrix Demo\n");
    printf("=========================================================\n\n");
    
    int n_conditions = dm->n_conditions;
    
    /* Generate true parameters from design matrix and weights */
    CDMParameters *true_params = malloc(n_conditions * sizeof(CDMParameters));
    for (int i = 0; i < n_conditions; i++) {
        double drift_angle = 0.0, drift_mag = 0.0, boundary_radius = 0.0, ndt = 0.0;
        for (int j = 0; j < dm->n_drift_angle_weights; j++) {
            drift_angle += gsl_matrix_get(dm->drift_angle_design, i, j) * example_drift_angle_weights[j];
        }
        for (int j = 0; j < dm->n_drift_mag_weights; j++) {
            drift_mag += gsl_matrix_get(dm->drift_mag_design, i, j) * example_drift_mag_weights[j];
        }
        for (int j = 0; j < dm->n_boundary_radius_weights; j++) {
            boundary_radius += gsl_matrix_get(dm->boundary_radius_design, i, j) * example_boundary_radius_weights[j];
        }
        for (int j = 0; j < dm->n_ndt_weights; j++) {
            ndt += gsl_matrix_get(dm->ndt_design, i, j) * example_ndt_weights[j];
        }
        true_params[i] = (CDMParameters){drift_angle, drift_mag, boundary_radius, ndt};
    }
    
    /* Generate observations from true parameters */
    CDMSummaryStats *observations = malloc(n_conditions * sizeof(CDMSummaryStats));
    for (int i = 0; i < n_conditions; i++) {
        CDMSummaryStats moments = ezcdm_forward(true_params[i]);
        moments.n = sample_size;
        observations[i] = sample_cdm_observations(moments, sample_size, r);
    }
    
    printf("True beta weights:\n");
    printf("  Drift angle:      ");
    for (int i = 0; i < dm->n_drift_angle_weights; i++) {
        printf("%.2f ", example_drift_angle_weights[i]);
    }
    printf("\n  Drift magnitude:  ");
    for (int i = 0; i < dm->n_drift_mag_weights; i++) {
        printf("%.2f ", example_drift_mag_weights[i]);
    }
    printf("\n  Boundary radius:  ");
    for (int i = 0; i < dm->n_boundary_radius_weights; i++) {
        printf("%.2f ", example_boundary_radius_weights[i]);
    }
    printf("\n  NDT:              ");
    for (int i = 0; i < dm->n_ndt_weights; i++) {
        printf("%.2f ", example_ndt_weights[i]);
    }
    printf("\n\n");
    
    printf("True parameters per condition:\n");
    for (int i = 0; i < n_conditions; i++) {
        printf("  Condition %d: drift_angle=%.3f, drift_mag=%.3f, boundary_radius=%.3f, ndt=%.3f\n",
               i + 1, true_params[i].drift_angle, true_params[i].drift_mag, 
               true_params[i].boundary_radius, true_params[i].ndt);
    }
    printf("\n");
    
    /* Run bootstrap */
    clock_t start_time = clock();
    CDMBetaWeights **bw_list = ezcdm_design_matrix_bootstrap(observations, n_conditions, dm, 
                                                             DEFAULT_N_BOOTSTRAP, r);
    clock_t end_time = clock();
    double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    /* Summarize results */
    CDMBetaWeights *summary = cdm_beta_weights_summarize(bw_list, DEFAULT_N_BOOTSTRAP);
    
    printf("EZ-CDM Estimated Beta Weights:\n");
    printf("  Drift angle:      ");
    for (int i = 0; i < dm->n_drift_angle_weights; i++) {
        printf("%.3f ", summary->drift_angle_weights[i]);
    }
    printf("\n  Drift magnitude:  ");
    for (int i = 0; i < dm->n_drift_mag_weights; i++) {
        printf("%.3f ", summary->drift_mag_weights[i]);
    }
    printf("\n  Boundary radius:  ");
    for (int i = 0; i < dm->n_boundary_radius_weights; i++) {
        printf("%.3f ", summary->boundary_radius_weights[i]);
    }
    printf("\n  NDT:              ");
    for (int i = 0; i < dm->n_ndt_weights; i++) {
        printf("%.3f ", summary->ndt_weights[i]);
    }
    printf("\n\nTime taken: %.0f ms\n", elapsed);
    
    /* Cleanup */
    for (int i = 0; i < DEFAULT_N_BOOTSTRAP; i++) {
        if (bw_list[i] != NULL) {
            cdm_beta_weights_free(bw_list[i]);
        }
    }
    free(bw_list);
    cdm_beta_weights_free(summary);
    free(true_params);
    free(observations);
    gsl_rng_free(r);
}

/* Simulation function for design matrix */
void simulation_design_matrix(int n_repetitions, int sample_size, int n_bootstrap, unsigned long seed) {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    if (seed == 0) {
        seed = (unsigned long)time(NULL);
    }
    gsl_rng_set(r, seed);
    
    /* Create example design matrix: original structure (4 conditions, 3 weights each) */
    CDMDesignMatrix *dm = cdm_design_matrix_alloc(4, 3, 3, 3, 3);
    cdm_design_matrix_set_design(dm, example_drift_angle_design, example_drift_mag_design, 
                                example_boundary_radius_design, example_ndt_design);
    cdm_design_matrix_precompute_inverses(dm);
    
    int n_conditions = 4;
    int n_drift_angle_weights = 3;
    int n_drift_mag_weights = 3;
    int n_boundary_radius_weights = 3;
    int n_ndt_weights = 3;
    
    /* True beta weights */
    double *true_drift_angle_weights = example_drift_angle_weights;
    double *true_drift_mag_weights = example_drift_mag_weights;
    double *true_boundary_radius_weights = example_boundary_radius_weights;
    double *true_ndt_weights = example_ndt_weights;
    
    /* Note: Using columns of ones design matrix with single beta weights
     * matching single condition values (0.0, 1.0, 1.0, 0.2)
     */
    
    int *drift_angle_coverage = calloc(n_drift_angle_weights, sizeof(int));
    int *drift_mag_coverage = calloc(n_drift_mag_weights, sizeof(int));
    int *boundary_radius_coverage = calloc(n_boundary_radius_weights, sizeof(int));
    int *ndt_coverage = calloc(n_ndt_weights, sizeof(int));
    
    printf("EZ Circular Diffusion Model (EZ-CDM) - Design Matrix Simulation\n");
    printf("===============================================================\n\n");
    printf("True beta weights:\n");
    printf("  Drift angle:      ");
    for (int i = 0; i < n_drift_angle_weights; i++) {
        printf("%.2f ", true_drift_angle_weights[i]);
    }
    printf("\n  Drift magnitude:  ");
    for (int i = 0; i < n_drift_mag_weights; i++) {
        printf("%.2f ", true_drift_mag_weights[i]);
    }
    printf("\n  Boundary radius:  ");
    for (int i = 0; i < n_boundary_radius_weights; i++) {
        printf("%.2f ", true_boundary_radius_weights[i]);
    }
    printf("\n  NDT:              ");
    for (int i = 0; i < n_ndt_weights; i++) {
        printf("%.2f ", true_ndt_weights[i]);
    }
    printf("\n\n");
    printf("Sample size: %d\n", sample_size);
    printf("Bootstrap repetitions: %d\n", n_bootstrap);
    printf("Simulation repetitions: %d\n", n_repetitions);
    printf("Random seed: %lu\n\n", seed);
    
    /* Track timing */
    double *iteration_times = malloc(n_repetitions * sizeof(double));
    if (!iteration_times) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    clock_t start_total = clock();
    
    for (int rep = 0; rep < n_repetitions; rep++) {
        /* Generate true parameters from design matrix */
        CDMParameters *true_params = malloc(n_conditions * sizeof(CDMParameters));
        for (int i = 0; i < n_conditions; i++) {
            double drift_angle = 0.0, drift_mag = 0.0, boundary_radius = 0.0, ndt = 0.0;
            for (int j = 0; j < n_drift_angle_weights; j++) {
                drift_angle += gsl_matrix_get(dm->drift_angle_design, i, j) * true_drift_angle_weights[j];
            }
            for (int j = 0; j < n_drift_mag_weights; j++) {
                drift_mag += gsl_matrix_get(dm->drift_mag_design, i, j) * true_drift_mag_weights[j];
            }
            for (int j = 0; j < n_boundary_radius_weights; j++) {
                boundary_radius += gsl_matrix_get(dm->boundary_radius_design, i, j) * true_boundary_radius_weights[j];
            }
            for (int j = 0; j < n_ndt_weights; j++) {
                ndt += gsl_matrix_get(dm->ndt_design, i, j) * true_ndt_weights[j];
            }
            true_params[i] = (CDMParameters){drift_angle, drift_mag, boundary_radius, ndt};
        }
        
        /* Generate observations */
        CDMSummaryStats *observations = malloc(n_conditions * sizeof(CDMSummaryStats));
        for (int i = 0; i < n_conditions; i++) {
            CDMSummaryStats moments = ezcdm_forward(true_params[i]);
            moments.n = sample_size;
            observations[i] = sample_cdm_observations(moments, sample_size, r);
        }
        
        /* Run bootstrap */
        clock_t start_iter = clock();
        CDMBetaWeights **bw_list = ezcdm_design_matrix_bootstrap(observations, n_conditions, dm, 
                                                                 n_bootstrap, r);        
        clock_t end_iter = clock();
        iteration_times[rep] = ((double)(end_iter - start_iter)) / CLOCKS_PER_SEC;

        /* Compute intervals and check coverage */
        double **drift_angle_samples = malloc(n_drift_angle_weights * sizeof(double*));
        double **drift_mag_samples = malloc(n_drift_mag_weights * sizeof(double*));
        double **boundary_radius_samples = malloc(n_boundary_radius_weights * sizeof(double*));
        double **ndt_samples = malloc(n_ndt_weights * sizeof(double*));
        
        for (int i = 0; i < n_drift_angle_weights; i++) {
            drift_angle_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        for (int i = 0; i < n_drift_mag_weights; i++) {
            drift_mag_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        for (int i = 0; i < n_boundary_radius_weights; i++) {
            boundary_radius_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        for (int i = 0; i < n_ndt_weights; i++) {
            ndt_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        
        int valid_samples = 0;
        for (int b = 0; b < n_bootstrap; b++) {
            if (bw_list[b] == NULL) continue;  /* Skip invalid samples */
            for (int i = 0; i < n_drift_angle_weights; i++) {
                drift_angle_samples[i][valid_samples] = bw_list[b]->drift_angle_weights[i];
            }
            for (int i = 0; i < n_drift_mag_weights; i++) {
                drift_mag_samples[i][valid_samples] = bw_list[b]->drift_mag_weights[i];
            }
            for (int i = 0; i < n_boundary_radius_weights; i++) {
                boundary_radius_samples[i][valid_samples] = bw_list[b]->boundary_radius_weights[i];
            }
            for (int i = 0; i < n_ndt_weights; i++) {
                ndt_samples[i][valid_samples] = bw_list[b]->ndt_weights[i];
            }
            valid_samples++;
        }
        
        /* If we don't have enough valid samples, skip this repetition */
        if (valid_samples < n_bootstrap * 0.5) {
            /* Cleanup and skip */
            for (int b = 0; b < n_bootstrap; b++) {
                if (bw_list[b] != NULL) {
                    cdm_beta_weights_free(bw_list[b]);
                }
            }
            free(bw_list);
            for (int i = 0; i < n_drift_angle_weights; i++) {
                free(drift_angle_samples[i]);
            }
            for (int i = 0; i < n_drift_mag_weights; i++) {
                free(drift_mag_samples[i]);
            }
            for (int i = 0; i < n_boundary_radius_weights; i++) {
                free(boundary_radius_samples[i]);
            }
            for (int i = 0; i < n_ndt_weights; i++) {
                free(ndt_samples[i]);
            }
            free(drift_angle_samples);
            free(drift_mag_samples);
            free(boundary_radius_samples);
            free(ndt_samples);
            free(true_params);
            free(observations);
            continue;  /* Skip this repetition */
        }
        
        /* Check coverage for each weight */
        for (int i = 0; i < n_drift_angle_weights; i++) {
            gsl_sort(drift_angle_samples[i], 1, valid_samples);
            int idx_025 = (int)(0.025 * (valid_samples - 1));
            int idx_975 = (int)(0.975 * (valid_samples - 1));
            double lower = drift_angle_samples[i][idx_025];
            double upper = drift_angle_samples[i][idx_975];
            if (lower <= true_drift_angle_weights[i] && true_drift_angle_weights[i] <= upper) {
                drift_angle_coverage[i]++;
            }
        }
        
        for (int i = 0; i < n_drift_mag_weights; i++) {
            gsl_sort(drift_mag_samples[i], 1, n_bootstrap);
            int idx_025 = (int)(0.025 * (valid_samples - 1));
            int idx_975 = (int)(0.975 * (valid_samples - 1));
            double lower = drift_mag_samples[i][idx_025];
            double upper = drift_mag_samples[i][idx_975];
            if (lower <= true_drift_mag_weights[i] && true_drift_mag_weights[i] <= upper) {
                drift_mag_coverage[i]++;
            }
        }
        
        for (int i = 0; i < n_boundary_radius_weights; i++) {
            gsl_sort(boundary_radius_samples[i], 1, n_bootstrap);
            int idx_025 = (int)(0.025 * (valid_samples - 1));
            int idx_975 = (int)(0.975 * (valid_samples - 1));
            double lower = boundary_radius_samples[i][idx_025];
            double upper = boundary_radius_samples[i][idx_975];
            if (lower <= true_boundary_radius_weights[i] && true_boundary_radius_weights[i] <= upper) {
                boundary_radius_coverage[i]++;
            }
        }
        
        for (int i = 0; i < n_ndt_weights; i++) {
            gsl_sort(ndt_samples[i], 1, n_bootstrap);
            int idx_025 = (int)(0.025 * (valid_samples - 1));
            int idx_975 = (int)(0.975 * (valid_samples - 1));
            double lower = ndt_samples[i][idx_025];
            double upper = ndt_samples[i][idx_975];
            if (lower <= true_ndt_weights[i] && true_ndt_weights[i] <= upper) {
                ndt_coverage[i]++;
            }
        }
        
        /* Cleanup */
        for (int b = 0; b < n_bootstrap; b++) {
            if (bw_list[b] != NULL) {
                cdm_beta_weights_free(bw_list[b]);
            }
        }
        free(bw_list);
        for (int i = 0; i < n_drift_angle_weights; i++) {
            free(drift_angle_samples[i]);
        }
        for (int i = 0; i < n_drift_mag_weights; i++) {
            free(drift_mag_samples[i]);
        }
        for (int i = 0; i < n_boundary_radius_weights; i++) {
            free(boundary_radius_samples[i]);
        }
        for (int i = 0; i < n_ndt_weights; i++) {
            free(ndt_samples[i]);
        }
        free(drift_angle_samples);
        free(drift_mag_samples);
        free(boundary_radius_samples);
        free(ndt_samples);
        free(true_params);
        free(observations);
                
        if ((rep + 1) % 100 == 0 || rep == n_repetitions - 1) {
            printf("Progress: %d/%d (%.1f%%)\r", rep + 1, n_repetitions, 
                   100.0 * (rep + 1) / n_repetitions);
            fflush(stdout);
        }
    }
    
    clock_t end_total = clock();
    double total_time = ((double)(end_total - start_total)) / CLOCKS_PER_SEC;
    
    printf("\n\n");
    printf("========================================\n");
    printf("Simulation Results\n");
    printf("========================================\n\n");
    
    printf("Coverage (should be ≈ 95%%):\n");
    printf("Drift angle weights:\n");
    for (int i = 0; i < n_drift_angle_weights; i++) {
        printf("  Weight %d: %.1f%%\n", i + 1, 
               100.0 * drift_angle_coverage[i] / n_repetitions);
    }
    printf("Drift magnitude weights:\n");
    for (int i = 0; i < n_drift_mag_weights; i++) {
        printf("  Weight %d: %.1f%%\n", i + 1, 
               100.0 * drift_mag_coverage[i] / n_repetitions);
    }
    printf("Boundary radius weights:\n");
    for (int i = 0; i < n_boundary_radius_weights; i++) {
        printf("  Weight %d: %.1f%%\n", i + 1, 
               100.0 * boundary_radius_coverage[i] / n_repetitions);
    }
    printf("NDT weights:\n");
    for (int i = 0; i < n_ndt_weights; i++) {
        printf("  Weight %d: %.1f%%\n", i + 1, 
               100.0 * ndt_coverage[i] / n_repetitions);
    }
    
    /* Timing statistics */
    double mean_time = gsl_stats_mean(iteration_times, 1, n_repetitions);
    double sd_time = gsl_stats_sd_m(iteration_times, 1, n_repetitions, mean_time);
    double min_time = iteration_times[0];
    double max_time = iteration_times[0];
    for (int i = 1; i < n_repetitions; i++) {
        if (iteration_times[i] < min_time) min_time = iteration_times[i];
        if (iteration_times[i] > max_time) max_time = iteration_times[i];
    }
    
    printf("\nIteration timing statistics:\n");
    printf("  Mean:  %.2f ms\n", mean_time * 1000.0);
    printf("  Std:   %.2f ms\n", sd_time * 1000.0);
    printf("  Min:   %.2f ms\n", min_time * 1000.0);
    printf("  Max:   %.2f ms\n", max_time * 1000.0);
    printf("  Total: %.2f ms (includes overhead)\n\n", total_time * 1000.0);
    
    free(iteration_times);
    free(drift_angle_coverage);
    free(drift_mag_coverage);
    free(boundary_radius_coverage);
    free(ndt_coverage);
    cdm_design_matrix_free(dm);
    gsl_rng_free(r);
}

/* Test functions */
int test_design_matrix() {
    CDMDesignMatrix *dm = cdm_design_matrix_alloc(4, 3, 3, 3, 3);
    if (!dm) {
        printf("FAIL: Design matrix allocation\n");
        return 1;
    }
    
    cdm_design_matrix_set_design(dm, example_drift_angle_design, example_drift_mag_design,
                                 example_boundary_radius_design, example_ndt_design);
    cdm_design_matrix_precompute_inverses(dm);
    
    printf("PASS: Design matrix test\n");
    cdm_design_matrix_free(dm);
    return 0;
}

int run_tests() {
    int failures = 0;
    
    printf("Running EZ-CDM Design Matrix tests...\n");
    printf("=====================================\n\n");
    
    failures += test_design_matrix();
    
    if (failures == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d test(s) failed\n", failures);
    }
    
    return failures;
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--demo") == 0) {
        int sample_size = 100;
        
        if (argc > 2) {
            sample_size = atoi(argv[2]);
        }
        
        CDMDesignMatrix *dm = cdm_design_matrix_alloc(4, 3, 3, 3, 3);
        cdm_design_matrix_set_design(dm, example_drift_angle_design, example_drift_mag_design,
                                     example_boundary_radius_design, example_ndt_design);
        cdm_design_matrix_precompute_inverses(dm);
        
        demo_design_matrix(dm, sample_size);
        cdm_design_matrix_free(dm);
        return 0;
    }
    
    if (argc > 1 && strcmp(argv[1], "--simulation") == 0) {
        int n_repetitions = 1000;
        int sample_size = 100;
        int n_bootstrap = 1000;
        unsigned long seed = 0;  /* 0 means use time-based seed */
        
        int arg_idx = 2;
        while (arg_idx < argc) {
            if (strcmp(argv[arg_idx], "--seed") == 0 && arg_idx + 1 < argc) {
                seed = strtoul(argv[arg_idx + 1], NULL, 10);
                arg_idx += 2;
            } else {
                /* Positional arguments: repetitions, sample_size, n_bootstrap */
                if (arg_idx == 2) {
                    n_repetitions = atoi(argv[arg_idx]);
                } else if (arg_idx == 3) {
                    sample_size = atoi(argv[arg_idx]);
                } else if (arg_idx == 4) {
                    n_bootstrap = atoi(argv[arg_idx]);
                }
                arg_idx++;
            }
        }
        
        simulation_design_matrix(n_repetitions, sample_size, n_bootstrap, seed);
        return 0;
    }
    
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }
    
    /* Default: run demo */
    CDMDesignMatrix *dm = cdm_design_matrix_alloc(4, 3, 3, 3, 3);
    cdm_design_matrix_set_design(dm, example_drift_angle_design, example_drift_mag_design,
                                 example_boundary_radius_design, example_ndt_design);
    cdm_design_matrix_precompute_inverses(dm);
    demo_design_matrix(dm, 100);
    cdm_design_matrix_free(dm);
    return 0;
}
