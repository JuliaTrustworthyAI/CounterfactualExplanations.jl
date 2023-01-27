using CounterfactualExplanations
using MLJBase
using Random

"""
    load_blobs(n=100; seed=data_seed, kwrgs...)

Loads overlapping synthtetic data.
"""
function load_blobs(n=100; seed=data_seed, kwrgs...)

    Random.seed!(seed)

    X, y = make_blobs(n, 2; kwrgs...)
    counterfactual_data = CounterfactualData(X, y)

    return counterfactual_data

end

"""
    load_linearly_separable(n=100; seed=data_seed)

Loads linearly separable synthtetic data.
"""
function load_linearly_separable(n=100; seed=data_seed)

    counterfactual_data = load_blobs(
        n;
        seed=seed, centers=2, center_box=(-2 => 2), cluster_std=0.1
    )

    return counterfactual_data

end

"""
    load_overlapping(n=100; seed=data_seed)

Loads overlapping synthtetic data.
"""
function load_overlapping(n=100; seed=data_seed)

    counterfactual_data = load_blobs(
        n;
        seed=seed, centers=2, center_box=(-2 => 2), cluster_std=0.5
    )

    return counterfactual_data

end

"""
    load_multi_class(n=100; seed=data_seed)

Loads multi-class synthtetic data.
"""
function load_multi_class(n=100; seed=data_seed, centers=4)

    counterfactual_data = load_blobs(
        n;
        seed=seed, centers=centers, center_box=(-2 => 2), cluster_std=0.1
    )

    return counterfactual_data

end

"""
    load_circles(n=100; seed=data_seed, noise=0.15, factor=0.01)

Loads synthetic circles data.
"""
function load_circles(n=100; seed=data_seed, noise=0.15, factor=0.01)

    Random.seed!(seed)

    X, y = make_circles(n; noise=noise, factor=factor)
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
    counterfactual_data = CounterfactualData(X, y)

    return counterfactual_data

end

"""
    load_synthetic_data(n=100; seed=data_seed)

Loads all synthetic datasets and wraps them in a dictionary.
"""
function load_synthetic_data(n=100; seed=data_seed)
    data = Dict(
        :linearly_separable => load_linearly_separable(n; seed=seed),
        :overlapping => load_overlapping(n; seed=seed),
        :multi_class => load_multi_class(n; seed=seed),
        :circles => load_circles(n; seed=seed),
        :moons => load_moons(n; seed=seed),
    )
    return data
end

