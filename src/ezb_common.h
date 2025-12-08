/*
 * EZ Bootstrap (EZB) - Common definitions and functions
 * 
 * Shared structures and utility functions for EZ Bootstrap implementations.
 */

#ifndef EZB_COMMON_H
#define EZB_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>

/* Structure to hold summary statistics */
typedef struct {
    double accuracy;    /* p: proportion correct */
    double mean_rt;    /* m: mean response time */
    double var_rt;     /* s^2: variance of response time */
    int n;             /* sample size */
} SummaryStats;

/* Structure to hold parameters */
typedef struct {
    double boundary;   /* alpha: boundary separation */
    double drift;      /* delta: drift rate */
    double ndt;        /* tau: non-decision time */
} Parameters;

/* Structure to hold bootstrap results */
typedef struct {
    double *boundary;
    double *drift;
    double *ndt;
    int n_samples;
} BootstrapResults;

/* Forward declarations */
Parameters ez_inverse(SummaryStats stats);
SummaryStats ez_forward(Parameters params);
SummaryStats sample_observations(SummaryStats moments, int sample_size, const gsl_rng *r);
SummaryStats sample_bootstrap_stats(SummaryStats observed, const gsl_rng *r);
BootstrapResults bootstrap(SummaryStats observed, int n_bootstrap, const gsl_rng *r);
void print_summary(BootstrapResults results);
void free_bootstrap_results(BootstrapResults results);
int check_coverage(Parameters true_params, BootstrapResults results);

#endif /* EZB_COMMON_H */

