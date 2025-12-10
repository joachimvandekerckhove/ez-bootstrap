/*
 * EZ Bootstrap (EZB) - Design matrix implementation
 */

#include "ezb_design_matrix.h"
#include <math.h>

/* Allocate design matrix */
DesignMatrix* design_matrix_alloc(int n_conditions, 
                                  int n_boundary_weights,
                                  int n_drift_weights,
                                  int n_ndt_weights) {
    DesignMatrix *dm = malloc(sizeof(DesignMatrix));
    if (!dm) return NULL;
    
    dm->n_conditions = n_conditions;
    dm->n_boundary_weights = n_boundary_weights;
    dm->n_drift_weights = n_drift_weights;
    dm->n_ndt_weights = n_ndt_weights;
    
    dm->boundary_design = gsl_matrix_alloc(n_conditions, n_boundary_weights);
    dm->drift_design = gsl_matrix_alloc(n_conditions, n_drift_weights);
    dm->ndt_design = gsl_matrix_alloc(n_conditions, n_ndt_weights);
    
    dm->boundary_design_inv = NULL;
    dm->drift_design_inv = NULL;
    dm->ndt_design_inv = NULL;
    
    if (!dm->boundary_design || !dm->drift_design || !dm->ndt_design) {
        design_matrix_free(dm);
        return NULL;
    }
    
    return dm;
}

/* Free design matrix */
void design_matrix_free(DesignMatrix *dm) {
    if (!dm) return;
    
    if (dm->boundary_design) gsl_matrix_free(dm->boundary_design);
    if (dm->drift_design) gsl_matrix_free(dm->drift_design);
    if (dm->ndt_design) gsl_matrix_free(dm->ndt_design);
    if (dm->boundary_design_inv) gsl_matrix_free(dm->boundary_design_inv);
    if (dm->drift_design_inv) gsl_matrix_free(dm->drift_design_inv);
    if (dm->ndt_design_inv) gsl_matrix_free(dm->ndt_design_inv);
    
    free(dm);
}

/* Set design matrix data (row-major order) */
void design_matrix_set_design(DesignMatrix *dm,
                               double *boundary_design_data,
                               double *drift_design_data,
                               double *ndt_design_data) {
    int i, j;
    
    for (i = 0; i < dm->n_conditions; i++) {
        for (j = 0; j < dm->n_boundary_weights; j++) {
            gsl_matrix_set(dm->boundary_design, i, j, 
                          boundary_design_data[i * dm->n_boundary_weights + j]);
        }
        for (j = 0; j < dm->n_drift_weights; j++) {
            gsl_matrix_set(dm->drift_design, i, j, 
                          drift_design_data[i * dm->n_drift_weights + j]);
        }
        for (j = 0; j < dm->n_ndt_weights; j++) {
            gsl_matrix_set(dm->ndt_design, i, j, 
                          ndt_design_data[i * dm->n_ndt_weights + j]);
        }
    }
}

/* Precompute inverses for efficient bootstrap using QR decomposition */
int design_matrix_precompute_inverses(DesignMatrix *dm) {
    /* Boundary design inverse - compute column by column */
    if (dm->boundary_design_inv == NULL && dm->n_conditions >= dm->n_boundary_weights) {
        dm->boundary_design_inv = gsl_matrix_alloc(dm->n_boundary_weights, dm->n_conditions);
        if (dm->boundary_design_inv) {
            gsl_matrix *copy = gsl_matrix_alloc(dm->n_conditions, dm->n_boundary_weights);
            gsl_vector *tau = gsl_vector_alloc(dm->n_boundary_weights);
            gsl_vector *b = gsl_vector_alloc(dm->n_conditions);
            gsl_vector *x = gsl_vector_alloc(dm->n_boundary_weights);
            gsl_vector *residual = gsl_vector_alloc(dm->n_conditions);
            
            if (copy && tau && b && x && residual) {
                gsl_matrix_memcpy(copy, dm->boundary_design);
                gsl_linalg_QR_decomp(copy, tau);
                
                /* Solve for each column of identity matrix */
                for (int j = 0; j < dm->n_conditions; j++) {
                    gsl_vector_set_zero(b);
                    gsl_vector_set(b, j, 1.0);
                    gsl_linalg_QR_lssolve(copy, tau, b, x, residual);
                    
                    /* Store result as column j of inverse */
                    for (int i = 0; i < dm->n_boundary_weights; i++) {
                        gsl_matrix_set(dm->boundary_design_inv, i, j, gsl_vector_get(x, i));
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
    
    /* Drift design inverse */
    if (dm->drift_design_inv == NULL && dm->n_conditions >= dm->n_drift_weights) {
        dm->drift_design_inv = gsl_matrix_alloc(dm->n_drift_weights, dm->n_conditions);
        if (dm->drift_design_inv) {
            gsl_matrix *copy = gsl_matrix_alloc(dm->n_conditions, dm->n_drift_weights);
            gsl_vector *tau = gsl_vector_alloc(dm->n_drift_weights);
            gsl_vector *b = gsl_vector_alloc(dm->n_conditions);
            gsl_vector *x = gsl_vector_alloc(dm->n_drift_weights);
            gsl_vector *residual = gsl_vector_alloc(dm->n_conditions);
            
            if (copy && tau && b && x && residual) {
                gsl_matrix_memcpy(copy, dm->drift_design);
                gsl_linalg_QR_decomp(copy, tau);
                
                for (int j = 0; j < dm->n_conditions; j++) {
                    gsl_vector_set_zero(b);
                    gsl_vector_set(b, j, 1.0);
                    gsl_linalg_QR_lssolve(copy, tau, b, x, residual);
                    
                    for (int i = 0; i < dm->n_drift_weights; i++) {
                        gsl_matrix_set(dm->drift_design_inv, i, j, gsl_vector_get(x, i));
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
BetaWeights* design_matrix_estimate_weights(DesignMatrix *dm, Parameters *params, int n_conditions) {
    BetaWeights *bw = malloc(sizeof(BetaWeights));
    if (!bw) return NULL;
    
    bw->n_boundary_weights = dm->n_boundary_weights;
    bw->n_drift_weights = dm->n_drift_weights;
    bw->n_ndt_weights = dm->n_ndt_weights;
    
    bw->boundary_weights = malloc(dm->n_boundary_weights * sizeof(double));
    bw->drift_weights = malloc(dm->n_drift_weights * sizeof(double));
    bw->ndt_weights = malloc(dm->n_ndt_weights * sizeof(double));
    
    if (!bw->boundary_weights || !bw->drift_weights || !bw->ndt_weights) {
        beta_weights_free(bw);
        return NULL;
    }
    
    /* Create parameter vectors */
    gsl_vector *boundary_vec = gsl_vector_alloc(n_conditions);
    gsl_vector *drift_vec = gsl_vector_alloc(n_conditions);
    gsl_vector *ndt_vec = gsl_vector_alloc(n_conditions);
    
    for (int i = 0; i < n_conditions; i++) {
        gsl_vector_set(boundary_vec, i, params[i].boundary);
        gsl_vector_set(drift_vec, i, params[i].drift);
        gsl_vector_set(ndt_vec, i, params[i].ndt);
    }
    
    /* Use precomputed inverse if available (O(k^2) instead of O(k^3)) */
    if (dm->boundary_design_inv) {
        gsl_vector *result = gsl_vector_alloc(dm->n_boundary_weights);
        gsl_blas_dgemv(CblasNoTrans, 1.0, dm->boundary_design_inv, boundary_vec, 0.0, result);
        for (int i = 0; i < dm->n_boundary_weights; i++) {
            bw->boundary_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
    } else {
        /* Fallback to least squares */
        gsl_vector *result = gsl_vector_alloc(dm->n_boundary_weights);
        gsl_vector *tau = gsl_vector_alloc(dm->n_boundary_weights);
        gsl_matrix *copy = gsl_matrix_alloc(n_conditions, dm->n_boundary_weights);
        gsl_matrix_memcpy(copy, dm->boundary_design);
        gsl_linalg_QR_decomp(copy, tau);
        gsl_linalg_QR_lssolve(copy, tau, boundary_vec, result, NULL);
        for (int i = 0; i < dm->n_boundary_weights; i++) {
            bw->boundary_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
        gsl_vector_free(tau);
        gsl_matrix_free(copy);
    }
    
    /* Drift weights */
    if (dm->drift_design_inv) {
        gsl_vector *result = gsl_vector_alloc(dm->n_drift_weights);
        gsl_blas_dgemv(CblasNoTrans, 1.0, dm->drift_design_inv, drift_vec, 0.0, result);
        for (int i = 0; i < dm->n_drift_weights; i++) {
            bw->drift_weights[i] = gsl_vector_get(result, i);
        }
        gsl_vector_free(result);
    } else {
        gsl_vector *result = gsl_vector_alloc(dm->n_drift_weights);
        gsl_vector *tau = gsl_vector_alloc(dm->n_drift_weights);
        gsl_matrix *copy = gsl_matrix_alloc(n_conditions, dm->n_drift_weights);
        gsl_matrix_memcpy(copy, dm->drift_design);
        gsl_linalg_QR_decomp(copy, tau);
        gsl_linalg_QR_lssolve(copy, tau, drift_vec, result, NULL);
        for (int i = 0; i < dm->n_drift_weights; i++) {
            bw->drift_weights[i] = gsl_vector_get(result, i);
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
    
    gsl_vector_free(boundary_vec);
    gsl_vector_free(drift_vec);
    gsl_vector_free(ndt_vec);
    
    return bw;
}

/* Free beta weights */
void beta_weights_free(BetaWeights *bw) {
    if (!bw) return;
    free(bw->boundary_weights);
    free(bw->drift_weights);
    free(bw->ndt_weights);
    free(bw);
}

/* Summarize beta weights from bootstrap samples */
BetaWeights* beta_weights_summarize(BetaWeights **bw_list, int n_samples) {
    if (n_samples == 0 || !bw_list || !bw_list[0]) return NULL;
    
    BetaWeights *summary = malloc(sizeof(BetaWeights));
    summary->n_boundary_weights = bw_list[0]->n_boundary_weights;
    summary->n_drift_weights = bw_list[0]->n_drift_weights;
    summary->n_ndt_weights = bw_list[0]->n_ndt_weights;
    
    summary->boundary_weights = malloc(summary->n_boundary_weights * sizeof(double));
    summary->drift_weights = malloc(summary->n_drift_weights * sizeof(double));
    summary->ndt_weights = malloc(summary->n_ndt_weights * sizeof(double));
    
    /* Compute means, stds, and quantiles for each weight */
    /* Simplified - just compute means for now */
    for (int i = 0; i < summary->n_boundary_weights; i++) {
        double sum = 0.0;
        for (int j = 0; j < n_samples; j++) {
            sum += bw_list[j]->boundary_weights[i];
        }
        summary->boundary_weights[i] = sum / n_samples;
    }
    
    for (int i = 0; i < summary->n_drift_weights; i++) {
        double sum = 0.0;
        for (int j = 0; j < n_samples; j++) {
            sum += bw_list[j]->drift_weights[i];
        }
        summary->drift_weights[i] = sum / n_samples;
    }
    
    for (int i = 0; i < summary->n_ndt_weights; i++) {
        double sum = 0.0;
        for (int j = 0; j < n_samples; j++) {
            sum += bw_list[j]->ndt_weights[i];
        }
        summary->ndt_weights[i] = sum / n_samples;
    }
    
    return summary;
}

