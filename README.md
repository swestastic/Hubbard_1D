# 1D Hubbard Chain DQMC

## Overview

Two methods of modeling a periodic 1D Hubbard Chain using DQMC. Firstly using [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl) and secondly following a [writeup by Dr. Richard Scalettar](https://scalettar.physics.ucdavis.edu/michigan/howto1.pdf)

**Note** Hubbard_Chain.jl is intended to be used with [my fork of SmoQyDQMC](https://github.com/swestastic/SmoQyDQMC.jl/) which has significantly fewer file reads and writes.

The Interaction Hamiltonian for both methods is parameterized as follows in a particle-hole symmetric form:

```math
\hat{H}_U = \sum_{i,\nu} U_{\nu,i}(\hat{n}_{\uparrow,\nu,i}-\frac{1}{2})(\hat{n}_{\downarrow,\nu,i}-\frac{1}{2})
```
The Hopping Hamiltonian is written as follows:

```math
\hat{H}_K = -\sum_{ij\sigma}t_{ij}(c^\dagger_{i\sigma}c_{j\sigma}+c^\dagger_{j\sigma}c_{i\sigma})-\mu\sum_{i\sigma}n_{i\sigma}
```
Where the complete Hamiltonian is written as follows:

```math
\hat{H} = \hat{H}_U + \hat{H}_K
```

```math
\hat{H} = \sum_{i,\nu} U_{\nu,i}(\hat{n}_{\uparrow,\nu,i}-\frac{1}{2})(\hat{n}_{\downarrow,\nu,i}-\frac{1}{2}) -\sum_{ij\sigma}t_{ij}(c^\dagger_{i\sigma}c_{j\sigma}+c^\dagger_{j\sigma}c_{i\sigma})-\mu\sum_{i\sigma}n_{i\sigma}
```

## Files

- `Chain_Py_sarr.sh` and `Chain_SmoQy_sarr.sh` are used to run these simulations on a computer cluster running SLURM. Values to run for each simulation are contained in these bash files. They are set up to run 25 independent simulations each, which is set through the range of `SID` values.

- `Hubbard_Chain_DQMC.py` can be run from the command line using `python3 Hubbard_Chain_DQMC.py`. It accepts arguments through flags `--N`, `--U`, `--beta`, `--Mu`, and a few others. The output is a 8x1 NumPy array saved to a .txt file named according to the input parameters.

- `Hubbard_Chain.jl` accepts arguments similar to the Python version, but instead runs a SmoQyDQMC simulation of the system. Data outputs are in `global_stats.csv` which are stored in a directory named according to the input parameters.

- `save.py` is used to collect the data outputs from both simulations and output two text files, one for SmoQy and one for Python, named according to the input parameters.It also accepts command line arguments similar to the other files.

- `save.sh` is used to run `save.py` on a SLURM cluster. Values can be set in this file.

- `Plots.ipynb` generates plots for the densities and double occupancy as a function of $\mu$. 

- `Results` has some example data from a run with $\beta=2.0, U=4.0, N=50$

## Issues

- There is a minus sign flip somewhere related to $\mu$, needs to be addressed properly; currently has a temporary fix.

- Poor agreement at low chemical potential values and negative values.

- Likely some issues due to floating point precision and the matrix multiplications.
