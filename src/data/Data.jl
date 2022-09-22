module Data

using ..Models, ..Interoperability, ..GenerativeModels
include("functions.jl")
export load_synthetic, toy_data_linear, toy_data_multi, toy_data_non_linear,
    mnist_data, mnist_ensemble, mnist_model, mnist_vae,
    cats_dogs_data, cats_dogs_model

end