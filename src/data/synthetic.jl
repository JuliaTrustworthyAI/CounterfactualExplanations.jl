using CounterfactualExplanations
using MLJBase
using Random

"""
    load_linearly_separable(n=100; seed=data_seed)

Loads linearly separable synthtetic data.
"""
function load_linearly_separable(n=100; seed=data_seed)

    Random.seed!(seed)

    X, y = make_blobs(n, 2; centers=2, center_box=(-2 => 2), cluster_std=0.1)
    y .= y .== 2
    counterfactual_data = CounterfactualData(X, y)

    return counterfactual_data

end

"""
    load_overlapping(n=100; seed=data_seed)

Loads overlapping synthtetic data.
"""
function load_overlapping(n=100; seed=data_seed)

    Random.seed!(seed)

    X, y = make_blobs(n, 2; centers=2, center_box=(-2 => 2), cluster_std=0.5)
    y .= y .== 2
    counterfactual_data = CounterfactualData(X, y)

    return counterfactual_data

end

"""
    load_circles(n=100; seed=data_seed, noise=0.15, factor=0.01)

Loads synthetic circles data.
"""
function load_circles(n=100; seed=data_seed, noise=0.15, factor=0.01)

    Random.seed!(seed)

    X, y = make_circles(n; noise=noise, factor=factor)
    y .= y .== 2
    counterfactual_data = CounterfactualData(X, y)

    return counterfactual_data

end

"""
    load_moons(n=100; seed=data_seed, kwrgs...)

Loads synthetic moons data.
"""
function load_moons(n=100; seed=data_seed, kwrgs...)

    Random.seed!(seed)

    X, y = make_moons(n; kwrgs...)
    y .= y .== 2
    counterfactual_data = CounterfactualData(X, y)

    return counterfactual_data

end

