"""
    load_circles(n=100; seed=data_seed, noise=0.15, factor=0.01)

Loads synthetic circles data.
"""
function load_circles(n=100; seed=data_seed, noise=0.15, factor=0.01)
    Random.seed!(seed)

    X, y = make_circles(n; noise=noise, factor=factor)
    counterfactual_data = CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    return counterfactual_data
end
