"""
(Optional) Self-organising map on a 2D grid.

Implement:
    class SOM(grid_shape=(10, 10), n_iter=5000, lr0=0.5, sigma0=None, seed=None)
        prototypes: (rows * cols, d), initialised from random data points
        each step: sample x, find the best-matching unit b, then update every prototype:
            m_j += lr(t) * h_bj(t) * (x - m_j),  h_bj = exp(-grid_dist(b, j)^2 / (2 sigma(t)^2))
        lr and sigma decay exponentially over time
        quantisation_error(X), topographic_error(X), predict(X) -> index of the BMU

Notes:
    As sigma -> 0, the update becomes online k-means.
"""