"""
	AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"Base class for gradient-based counterfactual generators."
mutable struct GradientBasedGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}
    penalty::Union{Nothing,Function,Vector{Function}}
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}
    latent_space::Bool
    opt::Flux.Optimise.AbstractOptimiser
end

"""
	Generator(;
		loss::Union{Nothing,Function}=nothing,
		penalty::Union{Nothing,Function,Vector{Function}}=nothing,
		λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing,
		latent_space::Bool::false,
		opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
	)

Default outer constructor for `Generator`.

# Arguments
- `loss::Union{Nothing,Function}=nothing`: The loss function used by the model.
- `penalty::Union{Nothing,Function,Vector{Function}}=nothing`: A penalty function for the generator to penalize counterfactuals too far from the original point.
- `λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing`: The weight of the penalty function.
- `latent_space::Bool=false`: Whether to use the latent space of a generative model to generate counterfactuals.
- `opt::Flux.Optimise.AbstractOptimiser=Flux.Descent()`: The optimizer to use for the generator.

# Returns
- `generator::GradientBasedGenerator`: A gradient-based counterfactual generator.
"""
function GradientBasedGenerator(;
    loss::Union{Nothing,Function}=nothing,
    penalty::Union{Nothing,Function,Vector{Function}}=nothing,
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}=nothing,
    latent_space::Bool=false,
    opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
)
    return GradientBasedGenerator(loss, penalty, λ, latent_space, opt)
end
