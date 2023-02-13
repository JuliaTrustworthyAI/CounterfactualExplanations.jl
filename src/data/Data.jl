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
        :multi_class => load_multi_class,
    ),
    :tabular => Dict(
        :california_housing => load_california_housing,
        :credit_default => load_credit_default,
        :gmsc => load_gmsc,
    ),
    :vision => Dict(
        :mnist => load_mnist,
    ),
)

export data_catalogue
export load_linearly_separable, load_overlapping, load_multi_class
export load_blobs, load_circles, load_moons, load_multi_class, load_synthetic_data
export load_california_housing, load_credit_default, load_gmsc
export load_mnist, load_mnist_test

end
