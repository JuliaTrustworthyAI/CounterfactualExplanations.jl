"""
    load_overlapping(n=100; seed=Random.GLOBAL_RNG)

Loads overlapping synthetic data.
"""
function load_overlapping(n=100; seed=Random.GLOBAL_RNG)
    counterfactual_data = load_blobs(n; seed=seed, centers=2, cluster_std=2.0)

    return counterfactual_data
end
