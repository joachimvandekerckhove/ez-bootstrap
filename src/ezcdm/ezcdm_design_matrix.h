/*
 * EZ Circular Diffusion Model (EZ-CDM) - Design matrix structures
 */

#ifndef EZCDM_DESIGN_MATRIX_H
#define EZCDM_DESIGN_MATRIX_H

#include "ezcdm_common.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

/* Structure to hold beta weights */
typedef struct {
    double *drift_angle_weights;
    double *drift_mag_weights;
    double *boundary_radius_weights;
    double *ndt_weights;
    int n_drift_angle_weights;
    int n_drift_mag_weights;
    int n_boundary_radius_weights;
    int n_ndt_weights;
} CDMBetaWeights;

/* Structure to hold design matrix */
typedef struct {
    gsl_matrix *drift_angle_design;
    gsl_matrix *drift_mag_design;
    gsl_matrix *boundary_radius_design;
    gsl_matrix *ndt_design;
    gsl_matrix *drift_angle_design_inv;  /* Precomputed inverse for efficiency */
    gsl_matrix *drift_mag_design_inv;
    gsl_matrix *boundary_radius_design_inv;
    gsl_matrix *ndt_design_inv;
    int n_conditions;
    int n_drift_angle_weights;
    int n_drift_mag_weights;
    int n_boundary_radius_weights;
    int n_ndt_weights;
} CDMDesignMatrix;

/* Function declarations */
CDMDesignMatrix* cdm_design_matrix_alloc(int n_conditions, 
                                        int n_drift_angle_weights,
                                        int n_drift_mag_weights,
                                        int n_boundary_radius_weights,
                                        int n_ndt_weights);
void cdm_design_matrix_free(CDMDesignMatrix *dm);
void cdm_design_matrix_set_design(CDMDesignMatrix *dm,
                                   double *drift_angle_design_data,
                                   double *drift_mag_design_data,
                                   double *boundary_radius_design_data,
                                   double *ndt_design_data);
int cdm_design_matrix_precompute_inverses(CDMDesignMatrix *dm);
CDMBetaWeights* cdm_design_matrix_estimate_weights(CDMDesignMatrix *dm, CDMParameters *params, int n_conditions);
void cdm_beta_weights_free(CDMBetaWeights *bw);
CDMBetaWeights* cdm_beta_weights_summarize(CDMBetaWeights **bw_list, int n_samples);

#endif /* EZCDM_DESIGN_MATRIX_H */
