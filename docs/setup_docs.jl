setup_docs = quote

    using Pkg
    Pkg.activate("docs")

    using CounterfactualExplanations
    using CounterfactualExplanations: animate_path, counterfactual, counterfactual_label
    using CounterfactualExplanations.Models
    using CounterfactualExplanations.Data
    using Flux
    using Flux.Optimise: update!, Adam
    using Images
    using LinearAlgebra
    using Markdown
    using MLDatasets
    using MLDatasets: convert2image
    using MLJBase
    using MLJModels: OneHotEncoder
    using Plots
    using Random
    using StatsBase
    using Tables

    # Setup:
    theme(:wong)
    Random.seed!(2023)
    www_path = "docs/src/www"
    include("docs/src/utils.jl")
    synthetic = CounterfactualExplanations.Data.load_synthetic_data()

    # Counteractual data and model:
    counterfactual_data = load_linearly_separable()
    M = fit_model(counterfactual_data, :Linear)
    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
    x = select_factual(counterfactual_data, chosen)

end;
