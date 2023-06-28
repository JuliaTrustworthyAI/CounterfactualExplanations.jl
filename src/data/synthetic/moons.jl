"""
    load_moons(n=100; seed=Random.GLOBAL_RNG, kwrgs...)

Loads synthetic moons data.
"""
function load_moons(n=100; seed=Random.GLOBAL_RNG, kwrgs...)
    X, y = MLJBase.make_moons(n; rng=seed, kwrgs...)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    return counterfactual_data
end
