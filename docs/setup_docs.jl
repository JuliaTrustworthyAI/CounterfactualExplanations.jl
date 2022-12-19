setup_docs = quote

    using Pkg
    Pkg.activate("docs")

    using CounterfactualExplanations
    using Flux
    using Flux.Optimise: update!, Adam
    using LinearAlgebra
    using MLJBase
    using Plots
    using Random

    theme(:wong)
    www_path = "docs/src/generators/www/"
    Random.seed!(2023)

    include("docs/src/utils.jl")

end
