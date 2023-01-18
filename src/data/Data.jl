module Data

using CounterfactualExplanations
using ..GenerativeModels
using ..Models
using Random

const data_seed = 42

export load_synthetic, toy_data_linear, toy_data_multi, toy_data_non_linear
export mnist_data, mnist_ensemble, mnist_model, mnist_vae
export cats_dogs_data, cats_dogs_model

include("functions.jl")
include("synthetic.jl")
include("tabular.jl")
include("vision.jl")

const data_catalogue = Dict(
    :synthetic => Dict(
        :linearly_separable => load_linearly_separable(),
        :overlapping => load_overlapping(),
        :moons => load_moons(),
        :circles => load_circles(),
    ),
    :tabular => Dict(
        :california_housing => load_california_housing(),
        :credit_default => load_credit_default(),
        :gmsc => load_gmsc(),
    ),
    :vision => Dict(
        :mnist => load_mnist(),
    ),
)

export data_catalogue
export load_linearly_separable, load_overlapping, load_circles, load_moons
export load_california_housing, load_credit_default, load_gmsc
export load_mnist

end
