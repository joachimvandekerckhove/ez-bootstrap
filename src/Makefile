# Makefile for EZ Bootstrap implementations using GSL
# Builds both EZ-DDM (drift diffusion model) and EZ-CDM (circular diffusion model)

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lgsl -lgslcblas -lm

# EZ-DDM (Drift Diffusion Model) targets
EZDDM_DIR = ezddm
EZDDM_COMMON_SRC = $(EZDDM_DIR)/ezb_common.c
EZDDM_COMMON_OBJ = $(EZDDM_DIR)/ezb_common.o
EZDDM_TARGETS = $(EZDDM_DIR)/ezb_single $(EZDDM_DIR)/ezb_design_matrix $(EZDDM_DIR)/test_ezb

# EZ-CDM (Circular Diffusion Model) targets
EZCDM_DIR = ezcdm
EZCDM_COMMON_SRC = $(EZCDM_DIR)/ezcdm_common.c
EZCDM_COMMON_OBJ = $(EZCDM_DIR)/ezcdm_common.o
EZCDM_TARGETS = $(EZCDM_DIR)/ezcdm_single $(EZCDM_DIR)/ezcdm_design_matrix $(EZCDM_DIR)/test_ezcdm

# All targets
TARGETS = $(EZDDM_TARGETS) $(EZCDM_TARGETS)

all: $(TARGETS)

# EZ-DDM targets
$(EZDDM_DIR)/ezb_single: $(EZDDM_DIR)/ezb_single.c $(EZDDM_COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(EZDDM_COMMON_OBJ) $(LDFLAGS)

$(EZDDM_DIR)/ezb_design_matrix: $(EZDDM_DIR)/ezb_design_matrix.c $(EZDDM_DIR)/ezb_design_matrix_impl.c $(EZDDM_COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $(EZDDM_DIR)/ezb_design_matrix.c $(EZDDM_DIR)/ezb_design_matrix_impl.c $(EZDDM_COMMON_OBJ) $(LDFLAGS)

$(EZDDM_DIR)/test_ezb: $(EZDDM_DIR)/test_ezb.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(EZDDM_COMMON_OBJ): $(EZDDM_COMMON_SRC) $(EZDDM_DIR)/ezb_common.h
	$(CC) $(CFLAGS) -c -o $@ $(EZDDM_COMMON_SRC)

# EZ-CDM targets
$(EZCDM_DIR)/ezcdm_single: $(EZCDM_DIR)/ezcdm_single.c $(EZCDM_COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(EZCDM_COMMON_OBJ) $(LDFLAGS)

$(EZCDM_DIR)/ezcdm_design_matrix: $(EZCDM_DIR)/ezcdm_design_matrix.c $(EZCDM_DIR)/ezcdm_design_matrix_impl.c $(EZCDM_COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $(EZCDM_DIR)/ezcdm_design_matrix.c $(EZCDM_DIR)/ezcdm_design_matrix_impl.c $(EZCDM_COMMON_OBJ) $(LDFLAGS)

$(EZCDM_DIR)/test_ezcdm: $(EZCDM_DIR)/test_ezcdm.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(EZCDM_COMMON_OBJ): $(EZCDM_COMMON_SRC) $(EZCDM_DIR)/ezcdm_common.h
	$(CC) $(CFLAGS) -c -o $@ $(EZCDM_COMMON_SRC)

clean:
	rm -f $(TARGETS) $(EZDDM_COMMON_OBJ) $(EZCDM_COMMON_OBJ) $(EZDDM_DIR)/*.o $(EZCDM_DIR)/*.o

.PHONY: all clean
