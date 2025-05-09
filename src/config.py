# src/config.py

"""
Hyperparameter search space definitions for our tuning experiments.
"""

param_space = {
    "n_layers":    [2, 3, 4],             # maybe allow one more depth
    "units":       [32, 64, 128],         # drop 16 & 256
    "lr_min":      1e-3,                  # start at 1e-3
    "lr_max":      1e-2,                  # up to 1e-2
    "batch_sizes": [16, 32, 64],          # keep the three sizes
}
