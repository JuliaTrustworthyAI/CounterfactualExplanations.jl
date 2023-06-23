"""
    load_blobs(n=100; seed=Random.GLOBAL_RNG, kwrgs...)

Loads overlapping synthetic data.
"""
function load_blobs(n=100; seed=Random.GLOBAL_RNG, k=2, centers=2, kwrgs...)

    X, y = MLJBase.make_blobs(n, k; centers=centers, rng=seed, kwrgs...)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    return counterfactual_data
end
