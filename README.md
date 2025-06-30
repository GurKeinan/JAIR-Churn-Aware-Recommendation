# Modeling Churn in Recommender Systems with Aggregated Preferences - Simulations and Experiments

This repository contains the code for the paper "Modeling Churn in Recommender Systems with Aggregated Preferences".
Specifically, it contains the code for the simulations and experiments presented in the paper (section 6 and appendix A).

## Requirements

For the comparison of our Branch-and-Bound algorithm to the POMDPs baseline SARSOP, one needs to download the source code for SARSOP from the following link: [SARSOP](https://github.com/AdaCompNUS/sarsop).

Python packages required to run the code are:

- numpy 1.26.4
- scipy 1.13.0
- matplotlib 3.8.4
- pandas 2.0.0
- tqdm 4.65.0
- scikit-learn 1.0.1

## Towards running the code
One needs to compile both the SARSOP and the Branch-and-Bound algorithms before running the code.
- The SARSOP algorithm is compiled by running "make" command in the SARSOP directory:
```bash
cd sarsop/src
make
```
- The Branch-and-Bound algorithm is compiled by running the following command in the root directory:
```bash
g++ -o bnb_cpp bnb_cpp.cpp
```

## Running the code

- The synthetic simulations are run by executing the file `Modules/simulations.py`.
- The experiments on the MovieLens dataset are run by executing the file `movielens_simulations/compare_algorithms.py`.
- Plotting the belief walk in the 3D simplex is done by executing the file `plotting_belief_walks/branch_and_bound_plotting.py`.