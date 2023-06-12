"""
    load_moons(n=100; seed=data_seed, kwrgs...)

Loads synthetic moons data.
"""
function load_moons(n=100; seed=data_seed, kwrgs...)
    Random.seed!(seed)

    X, y = MLJBase.make_moons(n; kwrgs...)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(X, y)
    counterfactual_data.X = Float32.(counterfactual_data.X)

    return counterfactual_data
end
