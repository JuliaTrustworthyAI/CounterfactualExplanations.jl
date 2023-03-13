using Flux
using LinearAlgebra
using Parameters

"Class for Composable counterfactual generator following Wachter et al (2018)"
mutable struct ComposableGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}                       # loss function
    complexity::Union{Function,Vector{Function}}        # penalties
    λ::Union{AbstractFloat,Vector{AbstractFloat}}       # strength of penalties
    decision_threshold::Union{Nothing,AbstractFloat}    # probability threshold
    opt::Flux.Optimise.AbstractOptimiser                # optimizer
    τ::AbstractFloat                                    # tolerance for convergence
end

# API streamlining:
@with_kw struct ComposableGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
end
