/*
 * EZ Bootstrap (EZB) - Design matrix implementation
 * 
 * Implements EZ bootstrap for EZ diffusion models with design matrices.
 */

#include "ezb_common.h"
#include "ezb_design_matrix.h"

#define DEFAULT_N_BOOTSTRAP 1000

/* Example design matrix (4 conditions, 3 beta weights each) */
static double example_boundary_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    0, 0, 1
};

static double example_drift_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    0, 1, 0
};

static double example_ndt_design[] = {
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    1, 0, 0
};

static double example_boundary_weights[] = {1.0, 1.5, 2.0};
static double example_drift_weights[] = {0.4, 0.8, 1.2};
static double example_ndt_weights[] = {0.2, 0.3, 0.4};

/* EZB bootstrap for design matrix */
BetaWeights** ezb_design_matrix_bootstrap(SummaryStats *observations, int n_conditions,
                                          DesignMatrix *dm, int n_bootstrap, const gsl_rng *r) {
    BetaWeights **bw_list = malloc(n_bootstrap * sizeof(BetaWeights*));
    if (!bw_list) return NULL;
    
    /* Get initial parameters from observations */
    Parameters *params = malloc(n_conditions * sizeof(Parameters));
    for (int i = 0; i < n_conditions; i++) {
        params[i] = ez_inverse(observations[i]);
    }
    
    /* Bootstrap loop */
    for (int b = 0; b < n_bootstrap; b++) {
        /* Resample observations */
        SummaryStats *boot_obs = malloc(n_conditions * sizeof(SummaryStats));
        for (int i = 0; i < n_conditions; i++) {
            boot_obs[i] = sample_bootstrap_stats(observations[i], r);
        }
        
        /* Get parameters from resampled observations */
        Parameters *boot_params = malloc(n_conditions * sizeof(Parameters));
        for (int i = 0; i < n_conditions; i++) {
            boot_params[i] = ez_inverse(boot_obs[i]);
        }
        
        /* Estimate beta weights */
        bw_list[b] = design_matrix_estimate_weights(dm, boot_params, n_conditions);
        
        free(boot_obs);
        free(boot_params);
    }
    
    free(params);
    return bw_list;
}

/* Demo function for design matrix */
void demo_design_matrix(DesignMatrix *dm, int sample_size) {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long)time(NULL));
    
    printf("EZ Bootstrap (EZB) - Design Matrix Demo\n");
    printf("=======================================\n\n");
    
    int n_conditions = dm->n_conditions;
    
    /* Generate true parameters from design matrix and weights */
    Parameters *true_params = malloc(n_conditions * sizeof(Parameters));
    for (int i = 0; i < n_conditions; i++) {
        double boundary = 0.0, drift = 0.0, ndt = 0.0;
        for (int j = 0; j < dm->n_boundary_weights; j++) {
            boundary += gsl_matrix_get(dm->boundary_design, i, j) * example_boundary_weights[j];
        }
        for (int j = 0; j < dm->n_drift_weights; j++) {
            drift += gsl_matrix_get(dm->drift_design, i, j) * example_drift_weights[j];
        }
        for (int j = 0; j < dm->n_ndt_weights; j++) {
            ndt += gsl_matrix_get(dm->ndt_design, i, j) * example_ndt_weights[j];
        }
        true_params[i] = (Parameters){boundary, drift, ndt};
    }
    
    /* Generate observations from true parameters */
    SummaryStats *observations = malloc(n_conditions * sizeof(SummaryStats));
    for (int i = 0; i < n_conditions; i++) {
        SummaryStats moments = ez_forward(true_params[i]);
        moments.n = sample_size;
        observations[i] = sample_observations(moments, sample_size, r);
    }
    
    printf("True beta weights:\n");
    printf("  Boundary: ");
    for (int i = 0; i < dm->n_boundary_weights; i++) {
        printf("%.2f ", example_boundary_weights[i]);
    }
    printf("\n  Drift:    ");
    for (int i = 0; i < dm->n_drift_weights; i++) {
        printf("%.2f ", example_drift_weights[i]);
    }
    printf("\n  NDT:      ");
    for (int i = 0; i < dm->n_ndt_weights; i++) {
        printf("%.2f ", example_ndt_weights[i]);
    }
    printf("\n\n");
    
    printf("True parameters per condition:\n");
    for (int i = 0; i < n_conditions; i++) {
        printf("  Condition %d: boundary=%.3f, drift=%.3f, ndt=%.3f\n",
               i + 1, true_params[i].boundary, true_params[i].drift, true_params[i].ndt);
    }
    printf("\n");
    
    /* Run bootstrap */
    clock_t start_time = clock();
    BetaWeights **bw_list = ezb_design_matrix_bootstrap(observations, n_conditions, dm, 
                                                         DEFAULT_N_BOOTSTRAP, r);
    clock_t end_time = clock();
    double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    /* Summarize results */
    BetaWeights *summary = beta_weights_summarize(bw_list, DEFAULT_N_BOOTSTRAP);
    
    printf("EZB Estimated Beta Weights:\n");
    printf("  Boundary: ");
    for (int i = 0; i < dm->n_boundary_weights; i++) {
        printf("%.3f ", summary->boundary_weights[i]);
    }
    printf("\n  Drift:    ");
    for (int i = 0; i < dm->n_drift_weights; i++) {
        printf("%.3f ", summary->drift_weights[i]);
    }
    printf("\n  NDT:      ");
    for (int i = 0; i < dm->n_ndt_weights; i++) {
        printf("%.3f ", summary->ndt_weights[i]);
    }
    printf("\n\nTime taken: %.0f ms\n", elapsed);
    
    /* Cleanup */
    for (int i = 0; i < DEFAULT_N_BOOTSTRAP; i++) {
        beta_weights_free(bw_list[i]);
    }
    free(bw_list);
    beta_weights_free(summary);
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
    
    /* Create example design matrix */
    DesignMatrix *dm = design_matrix_alloc(4, 3, 3, 3);
    design_matrix_set_design(dm, example_boundary_design, example_drift_design, example_ndt_design);
    design_matrix_precompute_inverses(dm);
    
    int n_conditions = 4;
    int n_boundary_weights = 3;
    int n_drift_weights = 3;
    int n_ndt_weights = 3;
    
    /* True beta weights */
    double *true_boundary_weights = example_boundary_weights;
    double *true_drift_weights = example_drift_weights;
    double *true_ndt_weights = example_ndt_weights;
    
    int *boundary_coverage = calloc(n_boundary_weights, sizeof(int));
    int *drift_coverage = calloc(n_drift_weights, sizeof(int));
    int *ndt_coverage = calloc(n_ndt_weights, sizeof(int));
    
    printf("EZ Bootstrap (EZB) - Design Matrix Simulation\n");
    printf("=============================================\n\n");
    printf("True beta weights:\n");
    printf("  Boundary: ");
    for (int i = 0; i < n_boundary_weights; i++) {
        printf("%.2f ", true_boundary_weights[i]);
    }
    printf("\n  Drift:    ");
    for (int i = 0; i < n_drift_weights; i++) {
        printf("%.2f ", true_drift_weights[i]);
    }
    printf("\n  NDT:      ");
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
        Parameters *true_params = malloc(n_conditions * sizeof(Parameters));
        for (int i = 0; i < n_conditions; i++) {
            double boundary = 0.0, drift = 0.0, ndt = 0.0;
            for (int j = 0; j < n_boundary_weights; j++) {
                boundary += gsl_matrix_get(dm->boundary_design, i, j) * true_boundary_weights[j];
            }
            for (int j = 0; j < n_drift_weights; j++) {
                drift += gsl_matrix_get(dm->drift_design, i, j) * true_drift_weights[j];
            }
            for (int j = 0; j < n_ndt_weights; j++) {
                ndt += gsl_matrix_get(dm->ndt_design, i, j) * true_ndt_weights[j];
            }
            true_params[i] = (Parameters){boundary, drift, ndt};
        }
        
        /* Generate observations */
        SummaryStats *observations = malloc(n_conditions * sizeof(SummaryStats));
        for (int i = 0; i < n_conditions; i++) {
            SummaryStats moments = ez_forward(true_params[i]);
            moments.n = sample_size;
            observations[i] = sample_observations(moments, sample_size, r);
        }
        
        /* Run bootstrap */
        clock_t start_iter = clock();
        BetaWeights **bw_list = ezb_design_matrix_bootstrap(observations, n_conditions, dm, 
                                                             n_bootstrap, r);        
        clock_t end_iter = clock();
        iteration_times[rep] = ((double)(end_iter - start_iter)) / CLOCKS_PER_SEC;

        /* Compute intervals and check coverage */
        double **boundary_samples = malloc(n_boundary_weights * sizeof(double*));
        double **drift_samples = malloc(n_drift_weights * sizeof(double*));
        double **ndt_samples = malloc(n_ndt_weights * sizeof(double*));
        
        for (int i = 0; i < n_boundary_weights; i++) {
            boundary_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        for (int i = 0; i < n_drift_weights; i++) {
            drift_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        for (int i = 0; i < n_ndt_weights; i++) {
            ndt_samples[i] = malloc(n_bootstrap * sizeof(double));
        }
        
        for (int b = 0; b < n_bootstrap; b++) {
            for (int i = 0; i < n_boundary_weights; i++) {
                boundary_samples[i][b] = bw_list[b]->boundary_weights[i];
            }
            for (int i = 0; i < n_drift_weights; i++) {
                drift_samples[i][b] = bw_list[b]->drift_weights[i];
            }
            for (int i = 0; i < n_ndt_weights; i++) {
                ndt_samples[i][b] = bw_list[b]->ndt_weights[i];
            }
        }
        
        /* Check coverage for each weight */
        for (int i = 0; i < n_boundary_weights; i++) {
            gsl_sort(boundary_samples[i], 1, n_bootstrap);
            int idx_025 = (int)(0.025 * (n_bootstrap - 1));
            int idx_975 = (int)(0.975 * (n_bootstrap - 1));
            double lower = boundary_samples[i][idx_025];
            double upper = boundary_samples[i][idx_975];
            if (lower <= true_boundary_weights[i] && true_boundary_weights[i] <= upper) {
                boundary_coverage[i]++;
            }
        }
        
        for (int i = 0; i < n_drift_weights; i++) {
            gsl_sort(drift_samples[i], 1, n_bootstrap);
            int idx_025 = (int)(0.025 * (n_bootstrap - 1));
            int idx_975 = (int)(0.975 * (n_bootstrap - 1));
            double lower = drift_samples[i][idx_025];
            double upper = drift_samples[i][idx_975];
            if (lower <= true_drift_weights[i] && true_drift_weights[i] <= upper) {
                drift_coverage[i]++;
            }
        }
        
        for (int i = 0; i < n_ndt_weights; i++) {
            gsl_sort(ndt_samples[i], 1, n_bootstrap);
            int idx_025 = (int)(0.025 * (n_bootstrap - 1));
            int idx_975 = (int)(0.975 * (n_bootstrap - 1));
            double lower = ndt_samples[i][idx_025];
            double upper = ndt_samples[i][idx_975];
            if (lower <= true_ndt_weights[i] && true_ndt_weights[i] <= upper) {
                ndt_coverage[i]++;
            }
        }
        
        /* Cleanup */
        for (int b = 0; b < n_bootstrap; b++) {
            beta_weights_free(bw_list[b]);
        }
        free(bw_list);
        for (int i = 0; i < n_boundary_weights; i++) {
            free(boundary_samples[i]);
        }
        for (int i = 0; i < n_drift_weights; i++) {
            free(drift_samples[i]);
        }
        for (int i = 0; i < n_ndt_weights; i++) {
            free(ndt_samples[i]);
        }
        free(boundary_samples);
        free(drift_samples);
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
    printf("Boundary weights:\n");
    for (int i = 0; i < n_boundary_weights; i++) {
        printf("  Weight %d: %.1f%%\n", i + 1, 
               100.0 * boundary_coverage[i] / n_repetitions);
    }
    printf("Drift weights:\n");
    for (int i = 0; i < n_drift_weights; i++) {
        printf("  Weight %d: %.1f%%\n", i + 1, 
               100.0 * drift_coverage[i] / n_repetitions);
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
    free(boundary_coverage);
    free(drift_coverage);
    free(ndt_coverage);
    design_matrix_free(dm);
    gsl_rng_free(r);
}

/* Test functions */
int test_design_matrix() {
    DesignMatrix *dm = design_matrix_alloc(4, 3, 3, 3);
    if (!dm) {
        printf("FAIL: Design matrix allocation\n");
        return 1;
    }
    
    design_matrix_set_design(dm, example_boundary_design, example_drift_design, example_ndt_design);
    design_matrix_precompute_inverses(dm);
    
    printf("PASS: Design matrix test\n");
    design_matrix_free(dm);
    return 0;
}

int run_tests() {
    int failures = 0;
    
    printf("Running EZB Design Matrix tests...\n");
    printf("==================================\n\n");
    
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
        
        DesignMatrix *dm = design_matrix_alloc(4, 3, 3, 3);
        design_matrix_set_design(dm, example_boundary_design, example_drift_design, example_ndt_design);
        design_matrix_precompute_inverses(dm);
        
        demo_design_matrix(dm, sample_size);
        design_matrix_free(dm);
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
    DesignMatrix *dm = design_matrix_alloc(4, 3, 3, 3);
    design_matrix_set_design(dm, example_boundary_design, example_drift_design, example_ndt_design);
    design_matrix_precompute_inverses(dm);
    demo_design_matrix(dm, 100);
    design_matrix_free(dm);
    return 0;
}
