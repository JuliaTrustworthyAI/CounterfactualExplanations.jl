"""
    load_linearly_separable(n=100; seed=Random.GLOBAL_RNG)

Loads linearly separable synthetic data.
"""
function load_linearly_separable(n=100; seed=Random.GLOBAL_RNG)
    counterfactual_data = load_blobs(n; seed=seed, centers=2, cluster_std=0.5)

    return counterfactual_data
end
