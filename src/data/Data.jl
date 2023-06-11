module Data

using Random
using LazyArtifacts
using CounterfactualExplanations
using MLJBase
using CSV
using DataFrames
using MLJModels
using CounterfactualExplanations.DataPreprocessing
using Flux
using MLDatasets

const data_seed = 42
data_dir = joinpath(artifact"data-tabular", "data-tabular")

include("synthetic/blobs.jl")
include("synthetic/circles.jl")
include("synthetic/linearly_separable.jl")
include("synthetic/moons.jl")
include("synthetic/multi_class.jl")
include("synthetic/overlapping.jl")

include("tabular/california_housing.jl")
include("tabular/credit_default.jl")
include("tabular/gmsc.jl")
include("tabular/german_credit.jl")

include("vision/cifar_10.jl")
include("vision/fashion_mnist.jl")
include("vision/mnist.jl")

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
        :german_credit => load_german_credit,
    ),
    :vision => Dict(
        :mnist => load_mnist,
        :fashion_mnist => load_fashion_mnist,
        :cifar_10 => load_cifar_10,
    ),
)

"""
    load_synthetic_data(n=100; seed=data_seed)

Loads all synthetic datasets and wraps them in a dictionary.
"""
function load_synthetic_data(n=100; seed=data_seed, drop=nothing)
    _dict = data_catalogue[:synthetic]
    if !isnothing(drop)
        drop = drop isa Vector ? drop : [drop]
        @assert all(_drop in keys(_dict) for _drop in drop)
    else
        drop = []
    end
    _dict = filter(((k, v),) -> k ∉ [drop..., :blobs], _dict)
    data = Dict(key => fun(n; seed=seed) for (key, fun) in _dict)
    return data
end

"""
    load_tabular_data(n=nothing; drop=nothing)

Loads all tabular datasets and wraps them in a dictionary.
"""
function load_tabular_data(n=nothing; drop=nothing)
    _dict = data_catalogue[:tabular]
    if !isnothing(drop)
        drop = drop isa Vector ? drop : [drop]
        @assert all(_drop in keys(_dict) for _drop in drop)
    else
        drop = []
    end
    _dict = filter(((k, v),) -> k ∉ drop, _dict)
    data = Dict(key => fun(n) for (key, fun) in _dict)
    return data
end

export data_catalogue
export load_linearly_separable, load_overlapping, load_multi_class
export load_blobs, load_circles, load_moons, load_multi_class
export load_synthetic_data
export load_california_housing, load_credit_default, load_gmsc, load_german_credit
export load_tabular_data
export load_mnist, load_mnist_test
export load_fashion_mnist, load_fashion_mnist_test
export load_cifar_10, load_cifar_10_test

end
