/*
 * EZ Bootstrap (EZB) - Design matrix structures
 */

#ifndef EZB_DESIGN_MATRIX_H
#define EZB_DESIGN_MATRIX_H

#include "ezb_common.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/* Structure to hold beta weights */
typedef struct {
    double *boundary_weights;
    double *drift_weights;
    double *ndt_weights;
    int n_boundary_weights;
    int n_drift_weights;
    int n_ndt_weights;
} BetaWeights;

/* Structure to hold design matrix */
typedef struct {
    gsl_matrix *boundary_design;
    gsl_matrix *drift_design;
    gsl_matrix *ndt_design;
    gsl_matrix *boundary_design_inv;  /* Precomputed inverse for efficiency */
    gsl_matrix *drift_design_inv;
    gsl_matrix *ndt_design_inv;
    int n_conditions;
    int n_boundary_weights;
    int n_drift_weights;
    int n_ndt_weights;
} DesignMatrix;

/* Function declarations */
DesignMatrix* design_matrix_alloc(int n_conditions, 
                                  int n_boundary_weights,
                                  int n_drift_weights,
                                  int n_ndt_weights);
void design_matrix_free(DesignMatrix *dm);
void design_matrix_set_design(DesignMatrix *dm,
                               double *boundary_design_data,
                               double *drift_design_data,
                               double *ndt_design_data);
int design_matrix_precompute_inverses(DesignMatrix *dm);
BetaWeights* design_matrix_estimate_weights(DesignMatrix *dm, Parameters *params, int n_conditions);
void beta_weights_free(BetaWeights *bw);
BetaWeights* beta_weights_summarize(BetaWeights **bw_list, int n_samples);

#endif /* EZB_DESIGN_MATRIX_H */

