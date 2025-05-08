# src/config.py

"""
Hyperparameter search space definitions for our tuning experiments.
"""

param_space = {
    # number of hidden layers to try
    "n_layers":   [1, 2, 3],

    # number of units per hidden layer
    "units":      [16, 32, 64, 128],

    # learning‐rate range (we’ll sample log‐uniform between these)
    "lr_min":     1e-4,
    "lr_max":     1e-2,

    # batch sizes to try
    "batch_sizes":[16, 32, 64],
}
