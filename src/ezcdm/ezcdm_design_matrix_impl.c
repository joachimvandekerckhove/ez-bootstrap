/*
 * EZ Circular Diffusion Model (EZ-CDM) - Design matrix implementation
 */

#include "ezcdm_design_matrix.h"
#include <math.h>

/* Allocate design matrix */
CDMDesignMatrix* cdm_design_matrix_alloc(int n_conditions, 
                                        int n_drift_angle_weights,
                                        int n_drift_mag_weights,
                                        int n_boundary_radius_weights,
                                        int n_ndt_weights) {
    CDMDesignMatrix *dm = malloc(sizeof(CDMDesignMatrix));
    if (!dm) return NULL;
    
    dm->n_conditions = n_conditions;
    dm->n_drift_angle_weights = n_drift_angle_weights;
    dm->n_drift_mag_weights = n_drift_mag_weights;
    dm->n_boundary_radius_weights = n_boundary_radius_weights;
    dm->n_ndt_weights = n_ndt_weights;
    
    dm->drift_angle_design = gsl_matrix_alloc(n_conditions, n_drift_angle_weights);
    dm->drift_mag_design = gsl_matrix_alloc(n_conditions, n_drift_mag_weights);
    dm->boundary_radius_design = gsl_matrix_alloc(n_conditions, n_boundary_radius_weights);
    dm->ndt_design = gsl_matrix_alloc(n_conditions, n_ndt_weights);
    
    dm->drift_angle_design_inv = NULL;
    dm->drift_mag_design_inv = NULL;
    dm->boundary_radius_design_inv = NULL;
    dm->ndt_design_inv = NULL;
    
    if (!dm->drift_angle_design || !dm->drift_mag_design || 
        !dm->boundary_radius_design || !dm->ndt_design) {
        cdm_design_matrix_free(dm);
        return NULL;
    }
    
    return dm;
}

/* Free design matrix */
void cdm_design_matrix_free(CDMDesignMatrix *dm) {
    if (!dm) return;
    
    if (dm->drift_angle_design) gsl_matrix_free(dm->drift_angle_design);
    if (dm->drift_mag_design) gsl_matrix_free(dm->drift_mag_design);
    if (dm->boundary_radius_design) gsl_matrix_free(dm->boundary_radius_design);
    if (dm->ndt_design) gsl_matrix_free(dm->ndt_design);
    if (dm->drift_angle_design_inv) gsl_matrix_free(dm->drift_angle_design_inv);
    if (dm->drift_mag_design_inv) gsl_matrix_free(dm->drift_mag_design_inv);
    if (dm->boundary_radius_design_inv) gsl_matrix_free(dm->boundary_radius_design_inv);
    if (dm->ndt_design_inv) gsl_matrix_free(dm->ndt_design_inv);
    
    free(dm);
}

/* Set design matrix data (row-major order) */
void cdm_design_matrix_set_design(CDMDesignMatrix *dm,
                                   double *drift_angle_design_data,
                                   double *drift_mag_design_data,
                                   double *boundary_radius_design_data,
                                   double *ndt_design_data) {
    int i, j;
    
    for (i = 0; i < dm->n_conditions; i++) {
        for (j = 0; j < dm->n_drift_angle_weights; j++) {
            gsl_matrix_set(dm->drift_angle_design, i, j, 
                          drift_angle_design_data[i * dm->n_drift_angle_weights + j]);
        }
        for (j = 0; j < dm->n_drift_mag_weights; j++) {
            gsl_matrix_set(dm->drift_mag_design, i, j, 
                          drift_mag_design_data[i * dm->n_drift_mag_weights + j]);
        }
        for (j = 0; j < dm->n_boundary_radius_weights; j++) {
            gsl_matrix_set(dm->boundary_radius_design, i, j, 
                          boundary_radius_design_data[i * dm->n_boundary_radius_weights + j]);
        }
        for (j = 0; j < dm->n_ndt_weights; j++) {
            gsl_matrix_set(dm->ndt_design, i, j, 
                          ndt_design_data[i * dm->n_ndt_weights + j]);
        }
    }
}

/* Precompute inverses for efficient bootstrap using QR decomposition */
int cdm_design_matrix_precompute_inverses(CDMDesignMatrix *dm) {
    /* Drift angle design inverse */
    if (dm->drift_angle_design_inv == NULL && dm->n_conditions >= dm->n_drift_angle_weights) {
        dm->drift_angle_design_inv = gsl_matrix_alloc(dm->n_drift_angle_weights, dm->n_conditions);
        if (dm->drift_angle_design_inv) {
            gsl_matrix *copy = gsl_matrix_alloc(dm->n_conditions, dm->n_drift_angle_weights);
            gsl_vector *tau = gsl_vector_alloc(dm->n_drift_angle_weights);
            gsl_vector *b = gsl_vector_alloc(dm->n_conditions);
            gsl_vector *x = gsl_vector_alloc(dm->n_drift_angle_weights);
            gsl_vector *residual = gsl_vector_alloc(dm->n_conditions);
            
            if (copy && tau && b && x && residual) {
                gsl_matrix_memcpy(copy, dm->drift_angle_design);
                gsl_linalg_QR_decomp(copy, tau);
                
                for (int j = 0; j < dm->n_conditions; j++) {
                    gsl_vector_set_zero(b);
                    gsl_vector_set(b, j, 1.0);
                    gsl_linalg_QR_lssolve(copy, tau, b, x, residual);
                    
                    for (int i = 0; i < dm->n_drift_angle_weights; i++) {
                        gsl_matrix_set(dm->drift_angle_design_inv, i, j, gsl_vector_get(x, i));
                    }
                }
                
                gsl_matrix_free(copy);
                gsl_vector_free(tau);
                gsl_vector_free(b);
                gsl_vector_free(x);
                gsl_vector_free(residual);
            }
        }
    }
    
    /* Drift magnitude design inverse */
    if (dm->drift_mag_design_inv == NULL && dm->n_conditions >= dm->n_drift_mag_weights) {
        dm->drift_mag_design_inv = gsl_matrix_alloc(dm->n_drift_mag_weights, dm->n_conditions);
        if (dm->drift_mag_design_inv) {
            gsl_matrix *copy = gsl_matrix_alloc(dm->n_conditions, dm->n_drift_mag_weights);
            gsl_vector *tau = gsl_vector_alloc(dm->n_drift_mag_weights);
            gsl_vector *b = gsl_vector_alloc(dm->n_conditions);
            gsl_vector *x = gsl_vector_alloc(dm->n_drift_mag_weights);
            gsl_vector *residual = gsl_vector_alloc(dm->n_conditions);
            
            if (copy && tau && b && x && residual) {
                gsl_matrix_memcpy(copy, dm->drift_mag_design);
                gsl_linalg_QR_decomp(copy, tau);
                
                for (int j = 0; j < dm->n_conditions; j++) {
                    gsl_vector_set_zero(b);
                    gsl_vector_set(b, j, 1.0);
                    gsl_linalg_QR_lssolve(copy, tau, b, x, residual);
                    
                    for (int i = 0; i < dm->n_drift_mag_weights; i++) {
                        gsl_matrix_set(dm->drift_mag_design_inv, i, j, gsl_vector_get(x, i));
                    }
                }
                
                gsl_matrix_free(copy);
                gsl_vector_free(tau);
                gsl_vector_free(b);
                gsl_vector_free(x);
                gsl_vector_free(residual);
            }
        }
    }
    
    /* Boundary radius design inverse */
    if (dm->boundary_radius_design_inv == NULL && dm->n_conditions >= dm->n_boundary_radius_weights) {
        dm->boundary_radius_design_inv = gsl_matrix_alloc(dm->n_boundary_radius_weights, dm->n_conditions);
        if (dm->boundary_radius_design_inv) {
            gsl_matrix *copy = gsl_matrix_alloc(dm->n_conditions, dm->n_boundary_radius_weights);
            gsl_vector *tau = gsl_vector_alloc(dm->n_boundary_radius_weights);
            gsl_vector *b = gsl_vector_alloc(dm->n_conditions);
            gsl_vector *x = gsl_vector_alloc(dm->n_boundary_radius_weights);
            gsl_vector *residual = gsl_vector_alloc(dm->n_conditions);
            
            if (copy && tau && b && x && residual) {
                gsl_matrix_memcpy(copy, dm->boundary_radius_design);
                gsl_linalg_QR_decomp(copy, tau);
                
                for (int j = 0; j < dm->n_conditions; j++) {
                    gsl_vector_set_zero(b);
                    gsl_vector_set(b, j, 1.0);
                    gsl_linalg_QR_lssolve(copy, tau, b, x, residual);
                    
                    for (int i = 0; i < dm->n_boundary_radius_weights; i++) {
                        gsl_matrix_set(dm->boundary_radius_design_inv, i, j, gsl_vector_get(x, i));
                    }
                }
                
                gsl_matrix_free(copy);
                gsl_vector_free(tau);
                gsl_vector_free(b);
                gsl_vector_free(x);
                gsl_vector_free(residual);
            }
        }
    }
    
    /* NDT design inverse */
    if (dm->ndt_design_inv == NULL && dm->n_conditions >= dm->n_ndt_weights) {
        dm->ndt_design_inv = gsl_matrix_alloc(dm->n_ndt_weights, dm->n_conditions);
        if (dm->ndt_design_inv) {
            gsl_matrix *copy = gsl_matrix_alloc(dm->n_conditions, dm->n_ndt_weights);
            gsl_vector *tau = gsl_vector_alloc(dm->n_ndt_weights);
            gsl_vector *b = gsl_vector_alloc(dm->n_conditions);
            gsl_vector *x = gsl_vector_alloc(dm->n_ndt_weights);
            gsl_vector *residual = gsl_vector_alloc(dm->n_conditions);
            
            if (copy && tau && b && x && residual) {
                gsl_matrix_memcpy(copy, dm->ndt_design);
                gsl_linalg_QR_decomp(copy, tau);
                
                for (int j = 0; j < dm->n_conditions; j++) {
                    gsl_vector_set_zero(b);
                    gsl_vector_set(b, j, 1.0);
                    gsl_linalg_QR_lssolve(copy, tau, b, x, residual);
                    
                    for (int i = 0; i < dm->n_ndt_weights; i++) {
                        gsl_matrix_set(dm->ndt_design_inv, i, j, gsl_vector_get(x, i));
                    }
                }
                
                gsl_matrix_free(copy);
                gsl_vector_free(tau);
                gsl_vector_free(b);
                gsl_vector_free(x);
                gsl_vector_free(residual);
            }
        }
    }
    
    return 0;
}

/* Estimate beta weights from parameters using linear regression */
CDMBetaWeights* cdm_design_matrix_estimate_weights(CDMDesignMatrix *dm, CDMParameters *params, int n_conditions) {
    CDMBetaWeights *bw = malloc(sizeof(CDMBetaWeights));
    if (!bw) return NULL;
    
    bw->n_drift_angle_weights = dm->n_drift_angle_weights;
    bw->n_drift_mag_weights = dm->n_drift_mag_weights;
    bw->n_boundary_radius_weights = dm->n_boundary_radius_weights;
    bw->n_ndt_weights = dm->n_ndt_weights;
    
    bw->drift_angle_weights = malloc(dm->n_drift_angle_weights * sizeof(double));
    bw->drift_mag_weights = malloc(dm->n_drift_mag_weights * sizeof(double));
    bw->boundary_radius_weights = malloc(dm->n_boundary_radius_weights * sizeof(double));
    bw->ndt_weights = malloc(dm->n_ndt_weights * sizeof(double));
    
    if (!bw->drift_angle_weights || !bw->drift_mag_weights || 
        !bw->boundary_radius_weights || !bw->ndt_weights) {
        cdm_beta_weights_free(bw);
        return NULL;
    }
    
    /* Create parameter vectors */
    gsl_vector *drift_angle_vec = gsl_vector_alloc(n_conditions);
    gsl_vector *drift_mag_vec = gsl_vector_alloc(n_conditions);
    gsl_vector *boundary_radius_vec = gsl_vector_alloc(n_conditions);
    gsl_vector *ndt_vec = gsl_vector_alloc(n_conditions);
    
    /* Validate parameters before using them */
    for (int i = 0; i < n_conditions; i++) {
        /* Check for invalid parameters */
        if (params[i].drift_mag <= 0.0 || params[i].boundary_radius <= 0.0 || 
            params[i].ndt < 0.0) {
            /* Invalid parameters - return NULL */
            cdm_beta_weights_free(bw);
            gsl_vector_free(drift_angle_vec);
            gsl_vector_free(drift_mag_vec);
            gsl_vector_free(boundary_radius_vec);
            gsl_vector_free(ndt_vec);
            return NULL;
        }
        gsl_vector_set(drift_angle_vec, i, params[i].drift_angle);
        gsl_vector_set(drift_mag_vec, i, params[i].drift_mag);
        gsl_vector_set(boundary_radius_vec, i, params[i].boundary_radius);
        gsl_vector_set(ndt_vec, i, params[i].ndt);
    }
    
    /* Use precomputed inverse if available (O(k^2) instead of O(k^3)) */
    /* Drift angle weights */
    if (dm->drift_angle_design_inv) {
        gsl_vector *result = gsl_vector_alloc(dm->n_drift_angle_weights);
        gsl_blas_dgemv(CblasNoTrans, 1.0, dm->drift_angle_design_inv, drift_angle_vec, 0.0, result);
        for (int i = 0; i < dm->n_drift_angle_weights; i++) {
            bw->drift_angle_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
    } else {
        gsl_vector *result = gsl_vector_alloc(dm->n_drift_angle_weights);
        gsl_vector *tau = gsl_vector_alloc(dm->n_drift_angle_weights);
        gsl_matrix *copy = gsl_matrix_alloc(n_conditions, dm->n_drift_angle_weights);
        gsl_matrix_memcpy(copy, dm->drift_angle_design);
        gsl_linalg_QR_decomp(copy, tau);
        gsl_linalg_QR_lssolve(copy, tau, drift_angle_vec, result, NULL);
        for (int i = 0; i < dm->n_drift_angle_weights; i++) {
            bw->drift_angle_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
        gsl_vector_free(tau);
        gsl_matrix_free(copy);
    }
    
    /* Drift magnitude weights */
    if (dm->drift_mag_design_inv) {
        gsl_vector *result = gsl_vector_alloc(dm->n_drift_mag_weights);
        gsl_blas_dgemv(CblasNoTrans, 1.0, dm->drift_mag_design_inv, drift_mag_vec, 0.0, result);
        for (int i = 0; i < dm->n_drift_mag_weights; i++) {
            bw->drift_mag_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
    } else {
        gsl_vector *result = gsl_vector_alloc(dm->n_drift_mag_weights);
        gsl_vector *tau = gsl_vector_alloc(dm->n_drift_mag_weights);
        gsl_matrix *copy = gsl_matrix_alloc(n_conditions, dm->n_drift_mag_weights);
        gsl_matrix_memcpy(copy, dm->drift_mag_design);
        gsl_linalg_QR_decomp(copy, tau);
        gsl_linalg_QR_lssolve(copy, tau, drift_mag_vec, result, NULL);
        for (int i = 0; i < dm->n_drift_mag_weights; i++) {
            bw->drift_mag_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
        gsl_vector_free(tau);
        gsl_matrix_free(copy);
    }
    
    /* Boundary radius weights */
    if (dm->boundary_radius_design_inv) {
        gsl_vector *result = gsl_vector_alloc(dm->n_boundary_radius_weights);
        gsl_blas_dgemv(CblasNoTrans, 1.0, dm->boundary_radius_design_inv, boundary_radius_vec, 0.0, result);
        for (int i = 0; i < dm->n_boundary_radius_weights; i++) {
            bw->boundary_radius_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
    } else {
        gsl_vector *result = gsl_vector_alloc(dm->n_boundary_radius_weights);
        gsl_vector *tau = gsl_vector_alloc(dm->n_boundary_radius_weights);
        gsl_matrix *copy = gsl_matrix_alloc(n_conditions, dm->n_boundary_radius_weights);
        gsl_matrix_memcpy(copy, dm->boundary_radius_design);
        gsl_linalg_QR_decomp(copy, tau);
        gsl_linalg_QR_lssolve(copy, tau, boundary_radius_vec, result, NULL);
        for (int i = 0; i < dm->n_boundary_radius_weights; i++) {
            bw->boundary_radius_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
        gsl_vector_free(tau);
        gsl_matrix_free(copy);
    }
    
    /* NDT weights */
    if (dm->ndt_design_inv) {
        gsl_vector *result = gsl_vector_alloc(dm->n_ndt_weights);
        gsl_blas_dgemv(CblasNoTrans, 1.0, dm->ndt_design_inv, ndt_vec, 0.0, result);
        for (int i = 0; i < dm->n_ndt_weights; i++) {
            bw->ndt_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
    } else {
        gsl_vector *result = gsl_vector_alloc(dm->n_ndt_weights);
        gsl_vector *tau = gsl_vector_alloc(dm->n_ndt_weights);
        gsl_matrix *copy = gsl_matrix_alloc(n_conditions, dm->n_ndt_weights);
        gsl_matrix_memcpy(copy, dm->ndt_design);
        gsl_linalg_QR_decomp(copy, tau);
        gsl_linalg_QR_lssolve(copy, tau, ndt_vec, result, NULL);
        for (int i = 0; i < dm->n_ndt_weights; i++) {
            bw->ndt_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
        gsl_vector_free(tau);
        gsl_matrix_free(copy);
    }
    
    gsl_vector_free(drift_angle_vec);
    gsl_vector_free(drift_mag_vec);
    gsl_vector_free(boundary_radius_vec);
    gsl_vector_free(ndt_vec);
    
    return bw;
}

/* Free beta weights */
void cdm_beta_weights_free(CDMBetaWeights *bw) {
    if (!bw) return;
    free(bw->drift_angle_weights);
    free(bw->drift_mag_weights);
    free(bw->boundary_radius_weights);
    free(bw->ndt_weights);
    free(bw);
}

/* Summarize beta weights from bootstrap samples */
CDMBetaWeights* cdm_beta_weights_summarize(CDMBetaWeights **bw_list, int n_samples) {
    if (n_samples == 0 || !bw_list) return NULL;
    
    /* Find first valid sample to get dimensions */
    int first_valid = -1;
    for (int j = 0; j < n_samples; j++) {
        if (bw_list[j] != NULL) {
            first_valid = j;
            break;
        }
    }
    if (first_valid < 0) return NULL;  /* No valid samples */
    
    CDMBetaWeights *summary = malloc(sizeof(CDMBetaWeights));
    summary->n_drift_angle_weights = bw_list[first_valid]->n_drift_angle_weights;
    summary->n_drift_mag_weights = bw_list[first_valid]->n_drift_mag_weights;
    summary->n_boundary_radius_weights = bw_list[first_valid]->n_boundary_radius_weights;
    summary->n_ndt_weights = bw_list[first_valid]->n_ndt_weights;
    
    summary->drift_angle_weights = malloc(summary->n_drift_angle_weights * sizeof(double));
    summary->drift_mag_weights = malloc(summary->n_drift_mag_weights * sizeof(double));
    summary->boundary_radius_weights = malloc(summary->n_boundary_radius_weights * sizeof(double));
    summary->ndt_weights = malloc(summary->n_ndt_weights * sizeof(double));
    
    /* Compute means for each weight, counting only valid samples */
    for (int i = 0; i < summary->n_drift_angle_weights; i++) {
        double sum = 0.0;
        int count = 0;
        for (int j = 0; j < n_samples; j++) {
            if (bw_list[j] != NULL) {
                sum += bw_list[j]->drift_angle_weights[i];
                count++;
            }
        }
        summary->drift_angle_weights[i] = (count > 0) ? sum / count : 0.0;
    }
    
    for (int i = 0; i < summary->n_drift_mag_weights; i++) {
        double sum = 0.0;
        int count = 0;
        for (int j = 0; j < n_samples; j++) {
            if (bw_list[j] != NULL) {
                sum += bw_list[j]->drift_mag_weights[i];
                count++;
            }
        }
        summary->drift_mag_weights[i] = (count > 0) ? sum / count : 0.0;
    }
    
    for (int i = 0; i < summary->n_boundary_radius_weights; i++) {
        double sum = 0.0;
        int count = 0;
        for (int j = 0; j < n_samples; j++) {
            if (bw_list[j] != NULL) {
                sum += bw_list[j]->boundary_radius_weights[i];
                count++;
            }
        }
        summary->boundary_radius_weights[i] = (count > 0) ? sum / count : 0.0;
    }
    
    for (int i = 0; i < summary->n_ndt_weights; i++) {
        double sum = 0.0;
        int count = 0;
        for (int j = 0; j < n_samples; j++) {
            if (bw_list[j] != NULL) {
                sum += bw_list[j]->ndt_weights[i];
                count++;
            }
        }
        summary->ndt_weights[i] = (count > 0) ? sum / count : 0.0;
    }
    
    return summary;
}
