# sympnet-sweep: Hyperparameter Sweeps for Symplectic Structure-Preserving Neural Networks

### Introduction:

This project aims to reproduce the results of a paper that presents a framework for constructing symplectic neural networks (SympNets) based on geometric integrators: these models learn $\phi_h$ that generates time-series data $(x(t_i), x(t_i + h))$ for $i = 1, \ldots, n_{\text{data}}$, where $\phi_h(x(t_i)) = x(t_i + h)$ is the unknown symplectic mapping.[^1]

So far, the code:

- Initializes a Ray cluster on a specified number of nodes and associated resources
- Uses the Ray Tune API to sweep over `search` parameters as defined in `configs/sweep.yaml`
- Saves the results to `results` in a Pandas dataframe for post-run processing

### Usage:

Sync to ensure project dependencies are installed:

```bash
uv sync
```

An example shell script is included with this code, the associated job can be scheduled via:

```bash
sbatch sweep_ex.sh
```

The default port number Ray uses for its dashboard is `:8265`. You may also need the head node's IP address, which is logged to stdout in the example script.

Live metrics for each experiment can be viewed via MLflow:

```bash
# the default port number for mlflow is :5000, noted here for clarity
uv run mlflow ui --backend-store-uri sqlite:///$HOME/sympnet-sweep/mlruns/mlflow.db --port 5000 &
```

This project assumes that the compute nodes are all connected to internet. It is often the case that SLURM clusters configure compute nodes without such access. Feel free to clone the code and revise accordingly.

[^1]: B. K. Tapley, [*Symplectic Neural Networks Based on Dynamical Systems*](https://arxiv.org/abs/2408.09821), arXiv:2408.09821, 2024.