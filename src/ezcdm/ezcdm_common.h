/*
 * EZ Circular Diffusion Model (EZ-CDM) - Common definitions and functions
 * 
 * Shared structures and utility functions for EZ-CDM Bootstrap implementations.
 */

#ifndef EZCDM_COMMON_H
#define EZCDM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <string.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sf_bessel.h>

/* Structure to hold summary statistics */
typedef struct {
    double mca;        /* MCA: circular mean of choice angles */
    double vca;        /* VCA: circular variance of choice angles */
    double mrt;        /* MRT: mean response time */
    double vrt;        /* VRT: response time variance */
    int n;             /* sample size */
} CDMSummaryStats;

/* Structure to hold parameters */
typedef struct {
    double drift_angle;      /* theta: angle of drift direction */
    double drift_mag;         /* mu: magnitude of drift */
    double boundary_radius;   /* r: decision criterion radius */
    double ndt;               /* tau: non-decision time */
} CDMParameters;

/* Structure to hold bootstrap results */
typedef struct {
    double *drift_angle;
    double *drift_mag;
    double *boundary_radius;
    double *ndt;
    int n_samples;
} CDMBootstrapResults;

/* Forward declarations */
CDMParameters ezcdm_inverse(CDMSummaryStats stats);
CDMSummaryStats ezcdm_forward(CDMParameters params);
CDMSummaryStats sample_cdm_observations(CDMSummaryStats moments, int sample_size, const gsl_rng *r);
CDMSummaryStats sample_cdm_bootstrap_stats(CDMSummaryStats observed, const gsl_rng *r);
CDMBootstrapResults cdm_bootstrap(CDMSummaryStats observed, int n_bootstrap, const gsl_rng *r);
void print_cdm_summary(CDMBootstrapResults results);
void free_cdm_bootstrap_results(CDMBootstrapResults results);
int check_cdm_coverage(CDMParameters true_params, CDMBootstrapResults results);

/* Helper function to solve for kappa from R = I1(kappa)/I0(kappa) */
double solve_kappa_from_R(double R);

#endif /* EZCDM_COMMON_H */
