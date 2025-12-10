# EZ Bootstrap (EZB) - C Implementation

C implementation of the EZ Bootstrap algorithm for uncertainty quantification in EZ diffusion models, using the GNU Scientific Library (GSL).

## Structure

The implementation follows the same structure as the Python `ezas/qnd` implementation:

- **`ezb_common.h` / `ezb_common.c`**: Shared structures and functions (forward/inverse equations, bootstrap, etc.)
- **`ezb_single.c`**: Single-condition EZ bootstrap implementation
- **`ezb_design_matrix.c`**: Design matrix EZ bootstrap implementation (simplified)
- **`test_ezb.c`**: Test runner that executes all tests
- **`run_simulations.c`**: Simulation runner that executes both simulation studies

## Building

```bash
make
```

This will build all executables:
- `ezb_single`
- `ezb_design_matrix`
- `test_ezb`
- `run_simulations`

## Usage

### EZB Single

```bash
# Run demo
./ezb_single --demo [sample_size]

# Run simulation study
./ezb_single --simulation [n_repetitions] [sample_size] [n_bootstrap] [--seed SEED]

# Run tests
./ezb_single --test
```

### EZB Design Matrix

```bash
# Run demo
./ezb_design_matrix --demo [sample_size]

# Run simulation study
./ezb_design_matrix --simulation [n_repetitions] [sample_size] [n_bootstrap] [--seed SEED]

# Run tests
./ezb_design_matrix --test
```

### Test Runner

```bash
# Run all tests
./test_ezb
```

## Reproducible example of main simulations for the paper

```bash
# Full simulation study (1000 repetitions) with fixed seed
./ezb_single --simulation 10000 100 1000 --seed 1
./ezb_design_matrix --simulation 10000 100 1000  --seed 1

```

## Seed Parameter

Both simulation functions support a `--seed SEED` flag for reproducible results:
- If `--seed` is not provided or seed is 0, a time-based seed is used (non-reproducible)
- If `--seed` is provided with a non-zero value, that seed is used (reproducible)
- The seed value is displayed in the simulation output

## Algorithm

The EZ Bootstrap procedure:

1. **For each bootstrap repetition $b = 1, \ldots, B$:**
   - Sample bootstrap summary statistics $S^*_b$ from their known sampling distributions:
     - $n\hat{p}^* \sim \text{Binomial}(n, \hat{p})$
     - $\hat{m}^* \sim \mathcal{N}(\hat{m}, \hat{s}^2/n)$
     - $\hat{s}^{2*} \sim \text{Gamma}((n-1)/2, 2\hat{s}^2/(n-1))$
   - Transform $S^*_b$ through the EZ inverse equations to obtain parameter estimates $\theta^*_b$

2. **Aggregate** the bootstrap parameter estimates to compute means, standard deviations, and credible intervals

## Dependencies

- GSL (GNU Scientific Library)
- Standard C library

## Notes

- The design matrix implementation uses GSL matrix operations for beta weight estimation
- Design matrix inverses are precomputed for efficient bootstrap (O(k^2) per iteration instead of O(k^3))
- All random number generation uses GSL's high-quality generators
