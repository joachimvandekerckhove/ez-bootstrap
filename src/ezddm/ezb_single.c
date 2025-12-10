/*
 * EZ Bootstrap (EZB) - Single condition implementation
 * 
 * Implements EZ bootstrap for single-condition EZ diffusion models.
 */

#include "ezb_common.h"

/* Default parameters for demo and simulation */
#define DEFAULT_BOUNDARY 1.0
#define DEFAULT_DRIFT 0.5
#define DEFAULT_NDT 0.2
#define DEFAULT_SAMPLE_SIZE 100
#define DEFAULT_N_BOOTSTRAP 1000

/* Demo function */
void demo(Parameters true_params, int sample_size) {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long)time(NULL));
    
    printf("EZ Bootstrap (EZB) - Single Condition Demo\n");
    printf("==========================================\n\n");
    
    /* Forward: generate moments from true parameters */
    SummaryStats moments = ez_forward(true_params);
    moments.n = sample_size;
    
    /* Sample observations from moments */
    SummaryStats observations = sample_observations(moments, sample_size, r);
    
    printf("True parameters:\n");
    printf("  Boundary: %.4f\n", true_params.boundary);
    printf("  Drift:    %.4f\n", true_params.drift);
    printf("  NDT:      %.4f\n", true_params.ndt);
    printf("\nMoments:\n");
    printf("  Accuracy: %.4f\n", moments.accuracy);
    printf("  Mean RT:  %.4f\n", moments.mean_rt);
    printf("  Var RT:   %.4f\n", moments.var_rt);
    printf("\nObservations:\n");
    printf("  Accuracy: %.4f\n", observations.accuracy);
    printf("  Mean RT:  %.4f\n", observations.mean_rt);
    printf("  Var RT:   %.4f\n", observations.var_rt);
    printf("  N:        %d\n", observations.n);
    printf("\n");
    
    /* Run bootstrap */
    clock_t start_time = clock();
    BootstrapResults results = bootstrap(observations, DEFAULT_N_BOOTSTRAP, r);
    clock_t end_time = clock();
    double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    /* Check coverage */
    int coverage = check_coverage(true_params, results);
    int boundary_covered = (coverage & 4) != 0;
    int drift_covered = (coverage & 2) != 0;
    int ndt_covered = (coverage & 1) != 0;
    
    printf("EZB Estimated Parameters:\n");
    print_summary(results);
    
    printf("\nCoverage check:\n");
    printf("  Boundary: %s\n", boundary_covered ? "COVERED" : "NOT COVERED");
    printf("  Drift:    %s\n", drift_covered ? "COVERED" : "NOT COVERED");
    printf("  NDT:      %s\n", ndt_covered ? "COVERED" : "NOT COVERED");
    printf("\nTime taken: %.0f ms\n", elapsed);
    
    free_bootstrap_results(results);
    gsl_rng_free(r);
}

/* Simulation function */
void simulation(int n_repetitions, int sample_size, int n_bootstrap, unsigned long seed) {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    if (seed == 0) {
        seed = (unsigned long)time(NULL);
    }
    gsl_rng_set(r, seed);
    
    /* True parameters */
    Parameters true_params = {
        .boundary = DEFAULT_BOUNDARY,
        .drift = DEFAULT_DRIFT,
        .ndt = DEFAULT_NDT
    };
    
    int boundary_coverage = 0;
    int drift_coverage = 0;
    int ndt_coverage = 0;
    int total_coverage = 0;
    
    /* Track timing */
    double *iteration_times = malloc(n_repetitions * sizeof(double));
    if (!iteration_times) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    clock_t start_total = clock();
    
    printf("EZ Bootstrap (EZB) - Single Condition Simulation\n");
    printf("================================================\n\n");
    printf("True parameters:\n"); 
    printf("    boundary=%.2f, drift=%.2f, ndt=%.2f\n", 
           true_params.boundary, true_params.drift, true_params.ndt);
    printf("Sample size: %d\n", sample_size);
    printf("Bootstrap repetitions: %d\n", n_bootstrap);
    printf("Simulation repetitions: %d\n", n_repetitions);
    printf("Random seed: %lu\n\n", seed);
    
    for (int i = 0; i < n_repetitions; i++) {
        /* Forward: generate moments from true parameters */
        SummaryStats moments = ez_forward(true_params);
        moments.n = sample_size;
        
        /* Sample observations from moments */
        SummaryStats observations = sample_observations(moments, sample_size, r);
        
        clock_t start_iter = clock();
        /* Run bootstrap */
        BootstrapResults results = bootstrap(observations, n_bootstrap, r);
        clock_t end_iter = clock();
        
        /* Check coverage */
        int coverage = check_coverage(true_params, results);
        if (coverage & 4) boundary_coverage++;
        if (coverage & 2) drift_coverage++;
        if (coverage & 1) ndt_coverage++;
        if (coverage == 7) total_coverage++;  /* All three covered */
        
        free_bootstrap_results(results);
        
        iteration_times[i] = ((double)(end_iter - start_iter)) / CLOCKS_PER_SEC;
        
        /* Progress indicator */
        if ((i + 1) % 100 == 0 || i == n_repetitions - 1) {
            printf("Progress: %d/%d (%.1f%%)\r", i + 1, n_repetitions, 
                   100.0 * (i + 1) / n_repetitions);
            fflush(stdout);
        }
    }
    
    clock_t end_total = clock();
    double total_time = ((double)(end_total - start_total)) / CLOCKS_PER_SEC;
    
    printf("\n\n");
    printf("========================================\n");
    printf("Simulation Results\n");
    printf("========================================\n\n");
    
    printf("Coverage:\n");
    printf("  Boundary: %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * boundary_coverage / n_repetitions);
    printf("  Drift:    %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * drift_coverage / n_repetitions);
    printf("  NDT:      %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * ndt_coverage / n_repetitions);
    printf("  Total:    %.1f%%  (should be ≈ 86%%)\n", 
           100.0 * total_coverage / n_repetitions);
    
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
    gsl_rng_free(r);
}

/* Test functions */
int test_forward_inverse() {
    Parameters params = {1.0, 0.5, 0.2};
    SummaryStats moments = ez_forward(params);
    
    /* Need to set n for inverse */
    moments.n = 100;
    Parameters recovered = ez_inverse(moments);
    
    double tol = 0.001;
    if (fabs(params.boundary - recovered.boundary) > tol ||
        fabs(params.drift - recovered.drift) > tol ||
        fabs(params.ndt - recovered.ndt) > tol) {
        printf("FAIL: Forward-inverse test\n");
        printf("  True: boundary=%.4f, drift=%.4f, ndt=%.4f\n", 
               params.boundary, params.drift, params.ndt);
        printf("  Recovered: boundary=%.4f, drift=%.4f, ndt=%.4f\n", 
               recovered.boundary, recovered.drift, recovered.ndt);
        return 1;
    }
    printf("PASS: Forward-inverse test\n");
    return 0;
}

int test_bootstrap() {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, 42);  /* Fixed seed for reproducibility */
    
    SummaryStats observed = {0.75, 0.5, 0.01, 100};
    BootstrapResults results = bootstrap(observed, 100, r);
    
    if (results.n_samples != 100) {
        printf("FAIL: Bootstrap test - wrong number of samples\n");
        free_bootstrap_results(results);
        gsl_rng_free(r);
        return 1;
    }
    
    /* Check that results are reasonable */
    if (results.boundary[0] <= 0 || results.drift[0] == 0 || results.ndt[0] < 0) {
        printf("FAIL: Bootstrap test - invalid parameter values\n");
        free_bootstrap_results(results);
        gsl_rng_free(r);
        return 1;
    }
    
    printf("PASS: Bootstrap test\n");
    free_bootstrap_results(results);
    gsl_rng_free(r);
    return 0;
}

int run_tests() {
    int failures = 0;
    
    printf("Running EZB Single tests...\n");
    printf("===========================\n\n");
    
    failures += test_forward_inverse();
    failures += test_bootstrap();
    
    if (failures == 0) {
        printf("\nAll tests passed!\n");
    } else {
        printf("\n%d test(s) failed\n", failures);
    }
    
    return failures;
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--demo") == 0) {
        Parameters params = {DEFAULT_BOUNDARY, DEFAULT_DRIFT, DEFAULT_NDT};
        int sample_size = DEFAULT_SAMPLE_SIZE;
        
        if (argc > 2) {
            sample_size = atoi(argv[2]);
        }
        
        demo(params, sample_size);
        return 0;
    }
    
    if (argc > 1 && strcmp(argv[1], "--simulation") == 0) {
        int n_repetitions = 1000;
        int sample_size = DEFAULT_SAMPLE_SIZE;
        int n_bootstrap = DEFAULT_N_BOOTSTRAP;
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
        
        simulation(n_repetitions, sample_size, n_bootstrap, seed);
        return 0;
    }
    
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }
    
    /* Default: run demo */
    Parameters params = {DEFAULT_BOUNDARY, DEFAULT_DRIFT, DEFAULT_NDT};
    demo(params, DEFAULT_SAMPLE_SIZE);
    return 0;
}

