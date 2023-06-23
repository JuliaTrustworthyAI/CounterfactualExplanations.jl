"""
    load_multi_class(n=100; seed=Random.GLOBAL_RNG)

Loads multi-class synthetic data.
"""
function load_multi_class(n=100; seed=Random.GLOBAL_RNG, centers=4)
    counterfactual_data = load_blobs(n; seed=seed, centers=centers, cluster_std=0.5)

    return counterfactual_data
end
