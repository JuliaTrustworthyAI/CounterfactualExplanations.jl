setup = quote
    using Pkg
    Pkg.activate("dev/artifacts")

    # Deps:
    using CounterfactualExplanations
    using CounterfactualExplanations.Models
    using Flux
    using JointEnergyModels
    using MLDatasets
    using MLJFlux
    using TaijaData

    # Utils
    include("$(pwd())/dev/artifacts/utils.jl")
end
