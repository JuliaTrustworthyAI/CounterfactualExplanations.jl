"""
    load_blobs(n=100; seed=data_seed, kwrgs...)

Loads overlapping synthetic data.
"""
function load_blobs(n=100; seed=data_seed, k=2, centers=2, kwrgs...)
    Random.seed!(seed)

    X, y = make_blobs(n, k; centers=centers, kwrgs...)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    return counterfactual_data
end
