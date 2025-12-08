# EZ Bootstrap (EZB)

C implementation of the EZ Bootstrap algorithm for uncertainty quantification in EZ diffusion models, using the GNU Scientific Library (GSL).

## Overview

This repository contains a high-performance C implementation of the computationally efficient bootstrap (CEB) method for the EZ diffusion model. The EZ Bootstrap method provides fast uncertainty quantification by operating directly on summary statistics rather than raw trial data, achieving 100-1000x speed improvements over full Bayesian methods while maintaining comparable statistical validity.

## Project Structure

- **`src/`**: C implementation of the EZ Bootstrap algorithm
  - See `src/README.md` for detailed documentation
- **`tex/`**: LaTeX source files for the associated paper
  - See `tex/README.md` for compilation instructions
- **`run_simulations.sh`**: Script to run simulation studies

## Quick Start

### Building

```bash
cd src
make
```

This builds all executables:
- `ezb_single`: Single-condition EZ bootstrap
- `ezb_design_matrix`: Design matrix EZ bootstrap
- `test_ezb`: Test runner
- `run_simulations`: Simulation runner

### Running Simulations

```bash
# Run the full simulation study
./run_simulations.sh

# Or run individual simulations
cd src
./ezb_single --simulation 10000 100 1000 --seed 1
./ezb_design_matrix --simulation 10000 100 1000 --seed 1
```

## Dependencies

- **GSL (GNU Scientific Library)**: Required for mathematical operations and random number generation
- **Standard C library**: For basic functionality
- **Make**: For building the project

### Installing GSL

On Ubuntu/Debian:
```bash
sudo apt-get install libgsl-dev
```

On macOS:
```bash
brew install gsl
```

## Documentation

- **Source code documentation**: See `src/README.md`
- **Paper/documentation**: See `tex/README.md` for compilation instructions

