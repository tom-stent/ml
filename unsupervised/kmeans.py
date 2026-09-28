"""k-means clustering (Lloyd's algorithm) with k-means++ initialisation.

Implement:
    kmeans_plus_plus(X, k, rng) -> initial centres
        first centre uniformly at random; each next centre sampled with probability
        proportional to D(x)^2, the squared distance to the nearest chosen centre
    class KMeans(k, n_init=10, max_iter=300, tol=1e-6, seed=None)
        fit(X): for each initialisation, alternate assignment (argmin of distances) and
            update (cluster means) until the centres move less than tol; keep the run
            with the lowest inertia
        attributes: cluster_centers_, labels_, inertia_, inertia_history_
        predict(X)

Notes:
    Handle empty clusters, e.g. by reinitialising to the point farthest from its centre.
    Vectorise: distances via utils.distances.sq_euclidean.
"""