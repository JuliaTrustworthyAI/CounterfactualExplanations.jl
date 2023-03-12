using LinearAlgebra
using Parameters

"Class for Gravitational counterfactual generator following Joshi et al (2019)."
mutable struct REVISEGenerator <: AbstractLatentSpaceGenerator
    loss::Union{Nothing,Function} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat} # probability threshold
    opt::Flux.Optimise.AbstractOptimiser # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
@with_kw struct REVISEGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
end

"""
    REVISEGenerator(;
        loss::Union{Nothing,Function} = nothing,
        complexity::Function = LinearAlgebra.norm,
        λ::AbstractFloat = 0.1,
        decision_threshold = 0.5,
        kwargs...,
    )

An outer constructor method that instantiates a REVISE generator.

# Examples
```julia-repl
generator = REVISEGenerator()
```
"""
function REVISEGenerator(;
    loss::Union{Nothing,Function}=nothing,
    complexity::Function=Objectives.distance_l2,
    λ::AbstractFloat=0.1,
    decision_threshold=0.5,
    kwargs...
)
    params = REVISEGeneratorParams(; kwargs...)
    REVISEGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ)
end
