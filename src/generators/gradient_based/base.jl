"""
	AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"Type union for acceptable argument types for the `penalty` field of `GradientBasedGenerator`."
const Penalty = Union{Nothing,<:PenaltyOrFun,Vector{<:PenaltyOrFun},Vector{<:Tuple}}

"Base class for gradient-based counterfactual generators."
mutable struct GradientBasedGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}
    penalty::Penalty
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}
    latent_space::Bool
    dim_reduction::Bool
    opt::Flux.Optimise.AbstractOptimiser
    generative_model_params::NamedTuple
end

"""
	GradientBasedGenerator(;
		loss::Union{Nothing,Function}=nothing,
		penalty::Penalty=nothing,
		λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing,
		latent_space::Bool::false,
		opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
        generative_model_params::NamedTuple=(;),
	)

Default outer constructor for `GradientBasedGenerator`.

# Arguments
- `loss::Union{Nothing,Function}=nothing`: The loss function used by the model.
- `penalty::Penalty=nothing`: A penalty function for the generator to penalize counterfactuals too far from the original point.
- `λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing`: The weight of the penalty function.
- `latent_space::Bool=false`: Whether to use the latent space of a generative model to generate counterfactuals.
- `opt::Flux.Optimise.AbstractOptimiser=Flux.Descent()`: The optimizer to use for the generator.
- `generative_model_params::NamedTuple`: The parameters of the generative model associated with the generator.

# Returns
- `generator::GradientBasedGenerator`: A gradient-based counterfactual generator.
"""
function GradientBasedGenerator(;
    loss::Union{Nothing,Function}=nothing,
    penalty::Penalty=nothing,
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}=nothing,
    latent_space::Bool=false,
    dim_reduction::Bool=false,
    opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
    generative_model_params::NamedTuple=(;),
)
    @assert !(isnothing(λ) && !isnothing(penalty)) "Penalty function(s) provided but no penalty weight(s) provided."
    @assert !(isnothing(λ) && !isnothing(penalty)) "Penalty weight(s) provided but no penalty function(s) provided."
    if typeof(penalty) <: Vector
        @assert length(λ) == length(penalty) || length(λ) == 1 "The number of penalty weights must match the number of penalty functions or be equal to one."
        length(λ) == 1 && (λ = fill(λ[1], length(penalty)))     # if only one penalty weight is provided, use it for all penalties
    end
    return GradientBasedGenerator(
        loss, penalty, λ, latent_space, dim_reduction, opt, generative_model_params
    )
end
