module Data

using CounterfactualExplanations
using ..GenerativeModels
using ..Models
using Random

const data_seed = 42

include("synthetic.jl")
include("tabular.jl")
include("vision.jl")

"A dictionary that provides an overview of the various benchmark datasets and the methods to load them."
const data_catalogue = Dict(
    :synthetic => Dict(
        :linearly_separable => load_linearly_separable,
        :overlapping => load_overlapping,
        :multi_class => load_multi_class,
        :blobs => load_blobs,
        :moons => load_moons,
        :circles => load_circles,
    ),
    :tabular => Dict(
        :california_housing => load_california_housing,
        :credit_default => load_credit_default,
        :gmsc => load_gmsc,
    ),
    :vision => Dict(:mnist => load_mnist),
)

"""
    load_synthetic_data(n=100; seed=data_seed)

Loads all synthetic datasets and wraps them in a dictionary.
"""
function load_synthetic_data(n=100; seed=data_seed)
    _dict = filter(((k,v),) -> k != :blobs, data_catalogue[:synthetic])
    data = Dict(key => fun(n; seed=seed) for (key,fun) in _dict)
    return data
end

"""
    load_tabular_data(n=nothing; seed=data_seed)

Loads all tabular datasets and wraps them in a dictionary.
"""
function load_tabular_data(n=nothing)
    data = Dict(key => fun(n) for (key,fun) in data_catalogue[:tabular])
    return data
end

export data_catalogue
export load_linearly_separable, load_overlapping, load_multi_class
export load_blobs, load_circles, load_moons, load_multi_class
export load_synthetic_data
export load_california_housing, load_credit_default, load_gmsc
export load_tabular_data
export load_mnist, load_mnist_test

end
