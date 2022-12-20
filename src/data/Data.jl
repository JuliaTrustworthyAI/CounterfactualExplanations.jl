module Data

using CounterfactualExplanations
using ..GenerativeModels
using ..Models

export load_synthetic, toy_data_linear, toy_data_multi, toy_data_non_linear
export mnist_data, mnist_ensemble, mnist_model, mnist_vae
export cats_dogs_data, cats_dogs_model

include("functions.jl")

end
