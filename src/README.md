# EZ Bootstrap - C Implementation

C implementation of the EZ Bootstrap algorithm for uncertainty quantification in EZ models, using the GNU Scientific Library (GSL).

This repository contains implementations for two EZ models:
- **EZ-DDM** (EZ Drift Diffusion Model) - in `ezddm/` subdirectory
- **EZ-CDM** (EZ Circular Diffusion Model) - in `ez-cdm/` subdirectory

## Structure

### EZ-DDM (Drift Diffusion Model)

The EZ-DDM implementation is located in the `ezddm/` subdirectory:

- **`ezb_common.h` / `ezb_common.c`**: Shared structures and functions (forward/inverse equations, bootstrap, etc.)
- **`ezb_single.c`**: Single-condition EZ bootstrap implementation
- **`ezb_design_matrix.c`**: Design matrix EZ bootstrap implementation
- **`ezb_design_matrix.h`**: Design matrix structures
- **`ezb_design_matrix_impl.c`**: Design matrix implementation
- **`test_ezb.c`**: Test runner that executes all tests

### EZ-CDM (Circular Diffusion Model)

The EZ-CDM implementation is located in the `ez-cdm/` subdirectory:

- **`ezcdm_common.h` / `ezcdm_common.c`**: Shared structures and functions (forward/inverse equations using Bessel functions, bootstrap, etc.)
- **`ezcdm_single.c`**: Single-condition EZ-CDM bootstrap implementation
- **`ezcdm_design_matrix.c`**: Design matrix EZ-CDM bootstrap implementation
- **`ezcdm_design_matrix.h`**: Design matrix structures
- **`ezcdm_design_matrix_impl.c`**: Design matrix implementation
- **`test_ezcdm.c`**: Test runner that executes all tests

## Building

```bash
make
```

This will build all executables for both implementations:

**EZ-DDM executables:**
- `ezddm/ezb_single`
- `ezddm/ezb_design_matrix`
- `ezddm/test_ezb`

**EZ-CDM executables:**
- `ez-cdm/ezcdm_single`
- `ez-cdm/ezcdm_design_matrix`
- `ez-cdm/test_ezcdm`

## Usage

### EZ-DDM Single Condition

```bash
# Run demo
./ezddm/ezb_single --demo [sample_size]

# Run simulation study
./ezddm/ezb_single --simulation [n_repetitions] [sample_size] [n_bootstrap] [--seed SEED]

# Run tests
./ezddm/ezb_single --test
```

### EZ-DDM Design Matrix

```bash
# Run demo
./ezddm/ezb_design_matrix --demo [sample_size]

# Run simulation study
./ezddm/ezb_design_matrix --simulation [n_repetitions] [sample_size] [n_bootstrap] [--seed SEED]

# Run tests
./ezddm/ezb_design_matrix --test
```

### EZ-CDM Single Condition

```bash
# Run demo
./ez-cdm/ezcdm_single --demo [sample_size]

# Run simulation study
./ez-cdm/ezcdm_single --simulation [n_repetitions] [sample_size] [n_bootstrap] [--seed SEED]

# Run tests
./ez-cdm/ezcdm_single --test
```

### EZ-CDM Design Matrix

```bash
# Run demo
./ez-cdm/ezcdm_design_matrix --demo [sample_size]

# Run simulation study
./ez-cdm/ezcdm_design_matrix --simulation [n_repetitions] [sample_size] [n_bootstrap] [--seed SEED]

# Run tests
./ez-cdm/ezcdm_design_matrix --test
```

### Test Runners

```bash
# Run all EZ-DDM tests
./ezddm/test_ezb

# Run all EZ-CDM tests
./ez-cdm/test_ezcdm
```

## Reproducible Examples

### EZ-DDM Simulations

```bash
# Full simulation study (10000 repetitions) with fixed seed
./ezddm/ezb_single --simulation 10000 100 1000 --seed 1
./ezddm/ezb_design_matrix --simulation 10000 100 1000 --seed 1
```

### EZ-CDM Simulations

```bash
# Full simulation study (10000 repetitions) with fixed seed
./ez-cdm/ezcdm_single --simulation 10000 100 1000 --seed 1
./ez-cdm/ezcdm_design_matrix --simulation 10000 100 1000 --seed 1
```

## Seed Parameter

All simulation functions support a `--seed SEED` flag for reproducible results:
- If `--seed` is not provided or seed is 0, a time-based seed is used (non-reproducible)
- If `--seed` is provided with a non-zero value, that seed is used (reproducible)
- The seed value is displayed in the simulation output

## Algorithm

The EZ Bootstrap procedure:

1. **For each bootstrap repetition $b = 1, \ldots, B$:**
   - Sample bootstrap summary statistics $S^*_b$ from their known sampling distributions
   - Transform $S^*_b$ through the EZ inverse equations to obtain parameter estimates $\theta^*_b$

2. **Aggregate** the bootstrap parameter estimates to compute means, standard deviations, and credible intervals

### EZ-DDM Summary Statistics

For the EZ-DDM, the summary statistics are:
- Accuracy rate: $n\hat{p}^* \sim \text{Binomial}(n, \hat{p})$
- Mean RT: $\hat{m}^* \sim \mathcal{N}(\hat{m}, \hat{s}^2/n)$
- RT variance: $\hat{s}^{2*} \sim \text{Gamma}((n-1)/2, 2\hat{s}^2/(n-1))$

### EZ-CDM Summary Statistics

For the EZ-CDM, the summary statistics are:
- Circular mean of choice angles (MCA): approximately normal
- Circular variance of choice angles (VCA): approximately normal
- Mean RT (MRT): approximately normal via CLT
- RT variance (VRT): approximately normal via CLT

The EZ-CDM implementation uses modified Bessel functions ($I_0$ and $I_1$) for the forward and inverse transformations, with Fisher's piecewise approximation for solving the inverse relationship.

## Model Parameters

### EZ-DDM Parameters
- **Boundary separation** ($\alpha$): decision threshold
- **Drift rate** ($\delta$): rate of evidence accumulation
- **Non-decision time** ($\tau$): time for non-decision processes

### EZ-CDM Parameters
- **Drift angle** ($\theta$): angle of drift direction
- **Drift magnitude** ($\mu$): magnitude of drift
- **Boundary radius** ($r$): decision criterion radius
- **Non-decision time** ($\tau$): time for non-decision processes

## Dependencies

- GSL (GNU Scientific Library) - required for random number generation, statistics, linear algebra, and Bessel functions
- Standard C library

## Notes

- The design matrix implementations use GSL matrix operations for beta weight estimation
- Design matrix inverses are precomputed for efficient bootstrap (O(k^2) per iteration instead of O(k^3))
- All random number generation uses GSL's high-quality generators
- The EZ-CDM implementation requires GSL's special functions library for Bessel function computations
