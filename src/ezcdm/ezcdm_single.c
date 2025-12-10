/*
 * EZ Circular Diffusion Model (EZ-CDM) - Single condition implementation
 * 
 * Implements EZ bootstrap for single-condition EZ circular diffusion models.
 */

#include "ezcdm_common.h"

/* Default parameters for demo and simulation */
#define DEFAULT_DRIFT_ANGLE 0.0
#define DEFAULT_DRIFT_MAG 1.0
#define DEFAULT_BOUNDARY_RADIUS 1.0
#define DEFAULT_NDT 0.2
#define DEFAULT_SAMPLE_SIZE 100
#define DEFAULT_N_BOOTSTRAP 1000

/* Demo function */
void demo(CDMParameters true_params, int sample_size) {
    gsl_rng *r;
    const gsl_rng_type *T;
    
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long)time(NULL));
    
    printf("EZ Circular Diffusion Model (EZ-CDM) - Single Condition Demo\n");
    printf("============================================================\n\n");
    
    /* Forward: generate moments from true parameters */
    CDMSummaryStats moments = ezcdm_forward(true_params);
    moments.n = sample_size;
    
    /* Sample observations from moments */
    CDMSummaryStats observations = sample_cdm_observations(moments, sample_size, r);
    
    printf("True parameters:\n");
    printf("  Drift angle:      %.4f\n", true_params.drift_angle);
    printf("  Drift magnitude:  %.4f\n", true_params.drift_mag);
    printf("  Boundary radius:  %.4f\n", true_params.boundary_radius);
    printf("  NDT:              %.4f\n", true_params.ndt);
    printf("\nMoments:\n");
    printf("  MCA: %.4f\n", moments.mca);
    printf("  VCA: %.4f\n", moments.vca);
    printf("  MRT: %.4f\n", moments.mrt);
    printf("  VRT: %.4f\n", moments.vrt);
    printf("\nObservations:\n");
    printf("  MCA: %.4f\n", observations.mca);
    printf("  VCA: %.4f\n", observations.vca);
    printf("  MRT: %.4f\n", observations.mrt);
    printf("  VRT: %.4f\n", observations.vrt);
    printf("  N:   %d\n", observations.n);
    printf("\n");
    
    /* Run bootstrap */
    clock_t start_time = clock();
    CDMBootstrapResults results = cdm_bootstrap(observations, DEFAULT_N_BOOTSTRAP, r);
    clock_t end_time = clock();
    double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    /* Check coverage */
    int coverage = check_cdm_coverage(true_params, results);
    int drift_angle_covered = (coverage & 8) != 0;
    int drift_mag_covered = (coverage & 4) != 0;
    int boundary_radius_covered = (coverage & 2) != 0;
    int ndt_covered = (coverage & 1) != 0;
    
    printf("EZ-CDM Estimated Parameters:\n");
    print_cdm_summary(results);
    
    printf("\nCoverage check:\n");
    printf("  Drift angle:      %s\n", drift_angle_covered ? "COVERED" : "NOT COVERED");
    printf("  Drift magnitude:  %s\n", drift_mag_covered ? "COVERED" : "NOT COVERED");
    printf("  Boundary radius:  %s\n", boundary_radius_covered ? "COVERED" : "NOT COVERED");
    printf("  NDT:              %s\n", ndt_covered ? "COVERED" : "NOT COVERED");
    printf("\nTime taken: %.0f ms\n", elapsed);
    
    free_cdm_bootstrap_results(results);
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
    CDMParameters true_params = {
        .drift_angle = DEFAULT_DRIFT_ANGLE,
        .drift_mag = DEFAULT_DRIFT_MAG,
        .boundary_radius = DEFAULT_BOUNDARY_RADIUS,
        .ndt = DEFAULT_NDT
    };
    
    int drift_angle_coverage = 0;
    int drift_mag_coverage = 0;
    int boundary_radius_coverage = 0;
    int ndt_coverage = 0;
    int total_coverage = 0;
    
    /* Track timing */
    double *iteration_times = malloc(n_repetitions * sizeof(double));
    if (!iteration_times) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }
    
    clock_t start_total = clock();
    
    printf("EZ Circular Diffusion Model (EZ-CDM) - Single Condition Simulation\n");
    printf("==================================================================\n\n");
    printf("True parameters:\n");
    printf("    drift_angle=%.2f, drift_mag=%.2f, boundary_radius=%.2f, ndt=%.2f\n", 
            true_params.drift_angle, true_params.drift_mag, 
            true_params.boundary_radius, true_params.ndt);
    printf("Sample size: %d\n", sample_size);
    printf("Bootstrap repetitions: %d\n", n_bootstrap);
    printf("Simulation repetitions: %d\n", n_repetitions);
    printf("Random seed: %lu\n\n", seed);
    
    for (int i = 0; i < n_repetitions; i++) {
        /* Forward: generate moments from true parameters */
        CDMSummaryStats moments = ezcdm_forward(true_params);
        moments.n = sample_size;
        
        /* Sample observations from moments */
        CDMSummaryStats observations = sample_cdm_observations(moments, sample_size, r);
        
        clock_t start_iter = clock();
        /* Run bootstrap */
        CDMBootstrapResults results = cdm_bootstrap(observations, n_bootstrap, r);
        clock_t end_iter = clock();
        
        /* Check coverage */
        int coverage = check_cdm_coverage(true_params, results);
        if (coverage & 8) drift_angle_coverage++;
        if (coverage & 4) drift_mag_coverage++;
        if (coverage & 2) boundary_radius_coverage++;
        if (coverage & 1) ndt_coverage++;
        if (coverage == 15) total_coverage++;  /* All four covered */
        
        free_cdm_bootstrap_results(results);
        
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
    printf("  Drift angle:      %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * drift_angle_coverage / n_repetitions);
    printf("  Drift magnitude:  %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * drift_mag_coverage / n_repetitions);
    printf("  Boundary radius:  %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * boundary_radius_coverage / n_repetitions);
    printf("  NDT:              %.1f%%  (should be ≈ 95%%)\n", 
           100.0 * ndt_coverage / n_repetitions);
    printf("  Total:            %.1f%%  (should be ≈ 81%%)\n", 
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
    CDMParameters params = {0.0, 1.0, 1.0, 0.2};
    CDMSummaryStats moments = ezcdm_forward(params);
    
    /* Need to set n for inverse */
    moments.n = 100;
    CDMParameters recovered = ezcdm_inverse(moments);
    
    double tol = 0.05;  /* Tolerance for circular model (accounts for numerical precision and approximations) */
    if (fabs(params.drift_angle - recovered.drift_angle) > tol ||
        fabs(params.drift_mag - recovered.drift_mag) > tol ||
        fabs(params.boundary_radius - recovered.boundary_radius) > tol ||
        fabs(params.ndt - recovered.ndt) > tol) {
        printf("FAIL: Forward-inverse test\n");
        printf("  True: drift_angle=%.4f, drift_mag=%.4f, boundary_radius=%.4f, ndt=%.4f\n", 
               params.drift_angle, params.drift_mag, params.boundary_radius, params.ndt);
        printf("  Recovered: drift_angle=%.4f, drift_mag=%.4f, boundary_radius=%.4f, ndt=%.4f\n", 
               recovered.drift_angle, recovered.drift_mag, recovered.boundary_radius, recovered.ndt);
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
    
    CDMSummaryStats observed = {0.0, 0.2, 0.5, 0.01, 100};
    CDMBootstrapResults results = cdm_bootstrap(observed, 100, r);
    
    if (results.n_samples != 100) {
        printf("FAIL: Bootstrap test - wrong number of samples\n");
        free_cdm_bootstrap_results(results);
        gsl_rng_free(r);
        return 1;
    }
    
    /* Check that results are reasonable */
    if (results.drift_mag[0] <= 0 || results.boundary_radius[0] <= 0 || results.ndt[0] < 0) {
        printf("FAIL: Bootstrap test - invalid parameter values\n");
        free_cdm_bootstrap_results(results);
        gsl_rng_free(r);
        return 1;
    }
    
    printf("PASS: Bootstrap test\n");
    free_cdm_bootstrap_results(results);
    gsl_rng_free(r);
    return 0;
}

int run_tests() {
    int failures = 0;
    
    printf("Running EZ-CDM Single tests...\n");
    printf("==============================\n\n");
    
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
        CDMParameters params = {DEFAULT_DRIFT_ANGLE, DEFAULT_DRIFT_MAG, 
                                DEFAULT_BOUNDARY_RADIUS, DEFAULT_NDT};
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
    CDMParameters params = {DEFAULT_DRIFT_ANGLE, DEFAULT_DRIFT_MAG, 
                            DEFAULT_BOUNDARY_RADIUS, DEFAULT_NDT};
    demo(params, DEFAULT_SAMPLE_SIZE);
    return 0;
}
