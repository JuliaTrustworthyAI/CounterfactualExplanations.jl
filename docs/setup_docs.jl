setup_docs = quote

    using Pkg
    Pkg.activate("docs")

    using CounterfactualExplanations
    using CounterfactualExplanations: animate_path, counterfactual, counterfactual_label
    using CounterfactualExplanations.Models
    using CounterfactualExplanations.Data
    using Flux
    using Flux.Optimise: update!, Adam
    using LinearAlgebra
    using MLJBase
    using MLJModels: OneHotEncoder
    using Plots
    using Random
    using StatsBase

    theme(:wong)
    www_path = "docs/src/generators/www/"
    Random.seed!(2023)

    include("docs/src/utils.jl")

    synthetic = CounterfactualExplanations.Data.load_synthetic_data()

end
