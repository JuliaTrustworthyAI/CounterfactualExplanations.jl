setup = quote
    using Pkg
    Pkg.activate("dev/artifacts")

    # Deps:
    using CounterfactualExplanations
    using CounterfactualExplanations.Models
    using CounterfactualExplanations.Data
    using CSV
    using Flux
    using Images
    using MLDatasets

    # Utils
    include("$(pwd())/dev/artifacts/utils.jl")
end
