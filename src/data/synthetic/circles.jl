"""
    load_circles(n=100; seed=Random.GLOBAL_RNG, noise=0.15, factor=0.01)

Loads synthetic circles data.
"""
function load_circles(n=100; seed=Random.GLOBAL_RNG, noise=0.15, factor=0.01)
    X, y = MLJBase.make_circles(n; rng=seed, noise=noise, factor=factor)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    return counterfactual_data
end
