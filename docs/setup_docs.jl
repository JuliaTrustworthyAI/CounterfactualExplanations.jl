setup_docs = quote
    using Pkg
    Pkg.activate("docs")

    using Chain: @chain
    using CounterfactualExplanations
    using CounterfactualExplanations: counterfactual, counterfactual_label
    using CounterfactualExplanations.Convergence
    using CounterfactualExplanations.DataPreprocessing
    using CounterfactualExplanations.Evaluation: benchmark
    using CounterfactualExplanations.Generators
    using CounterfactualExplanations.Models
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
    using SliceMap
    using StatsBase
    using Tables
    using TaijaData
    using TaijaParallel
    using TaijaParallel: @with_parallelizer
    using TaijaPlotting: animate_path
    using TaijaInteroperability
    using TaijaData

    # Setup:
    theme(:wong)
    Random.seed!(2022)
    www_path = "$(pwd())/docs/src/www"
    include("$(pwd())/docs/src/utils.jl")
    synthetic = TaijaData.load_synthetic_data()
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Counteractual data and model:
    counterfactual_data = CounterfactualData(load_linearly_separable()...)
    M = fit_model(counterfactual_data, :Linear)
    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
    x = select_factual(counterfactual_data, chosen)

    # Search:
    generator = Generators.GenericGenerator()
    ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
end
