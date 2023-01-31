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
    using Markdown
    using MLDatasets
    using MLJBase
    using MLJModels: OneHotEncoder
    using Plots
    using Random
    using StatsBase

    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "docs/src/www"
    include("docs/src/utils.jl")
    synthetic = CounterfactualExplanations.Data.load_synthetic_data()

end;
