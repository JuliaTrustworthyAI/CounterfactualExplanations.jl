"""
    load_moons(n=250; seed=data_seed, kwrgs...)

Loads synthetic moons data.
"""
function load_moons(n=250; seed=data_seed, kwrgs...)
    if isa(seed, Random.AbstractRNG)
        X, y = MLJBase.make_moons(n; rng=seed, kwrgs...)
    else
        Random.seed!(seed)
        X, y = MLJBase.make_moons(n; kwrgs...)
    end
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    return counterfactual_data
end
