"""
    load_circles(n=250; seed=data_seed, noise=0.15, factor=0.01)

Loads synthetic circles data.
"""
function load_circles(n=250; seed=data_seed, noise=0.15, factor=0.01)
    if isa(seed, Random.AbstractRNG)
        X, y = MLJBase.make_circles(n; rng=seed, noise=noise, factor=factor)
    else
        Random.seed!(seed)
        X, y = MLJBase.make_circles(n; noise=noise, factor=factor)
    end
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)
    counterfactual_data.standardize = true
    return counterfactual_data
end
