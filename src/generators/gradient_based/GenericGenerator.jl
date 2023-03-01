using Flux
using LinearAlgebra
using Parameters

"Class for Generic counterfactual generator following Wachter et al (2018)"
mutable struct GenericGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}                       # loss function
    complexity::Function                                # complexity function
    λ::AbstractFloat                                    # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat}    # probability threshold
    opt::Flux.Optimise.AbstractOptimiser                # optimizer
    τ::AbstractFloat                                    # tolerance for convergence
end

# API streamlining:
@with_kw struct GenericGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
end

"""
    GenericGenerator(;
        loss::Union{Nothing,Function}=nothing,
        complexity::Function=LinearAlgebra.norm,
        λ::AbstractFloat=0.1,
        decision_threshold=0.5,
        kwargs...
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = GenericGenerator()
```
"""
function GenericGenerator(;
    loss::Union{Nothing,Function}=nothing,
    complexity::Function=LinearAlgebra.norm,
    λ::AbstractFloat=0.1,
    decision_threshold=0.5,
    kwargs...
)
    params = GenericGeneratorParams(; kwargs...)
    GenericGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ)
end
