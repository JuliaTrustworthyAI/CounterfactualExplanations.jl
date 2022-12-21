setup_docs = quote

    using Pkg
    Pkg.activate("docs")

    using CounterfactualExplanations
    using CounterfactualExplanations: counterfactual, counterfactual_label
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

    synthetic = CounterfactualExplanations.Data.load_synthetic()

end
