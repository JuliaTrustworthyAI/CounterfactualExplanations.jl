using LinearAlgebra

# -------- Wachter et al (2018): 
struct GenericGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
using Parameters
@with_kw struct GenericGeneratorParams
    ϵ::AbstractFloat=0.1
    τ::AbstractFloat=1e-5
end

"""
    GenericGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = GenericGenerator()
```
"""
GenericGenerator(
    ;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Function=norm,
    λ::AbstractFloat=0.1,
    params::Union{NamedTuple,GenericGeneratorParams}=GenericGeneratorParams()
) = GenericGenerator(loss, complexity, λ, params.ϵ, params.τ)