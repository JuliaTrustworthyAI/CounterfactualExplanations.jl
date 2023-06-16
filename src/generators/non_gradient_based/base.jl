abstract type AbstractNonGradientBasedGenerator <: AbstractGenerator end

"Base class for heuristic-based counterfactual generators."
mutable struct HeuristicBasedGenerator <: AbstractNonGradientBasedGenerator
    penalty::Union{Nothing,Function,Vector{Function}}
    ϵ::Union{Nothing,AbstractFloat}
    latent_space::Bool
end

"""
    HeuristicBasedGenerator(;
        penalty::Union{Nothing,Function,Vector{Function}}=nothing,
        ϵ::Union{Nothing,AbstractFloat}=nothing,
    )

Default outer constructor for `HeuristicBasedGenerator`.

# Arguments
- `penalty::Union{Nothing,Function,Vector{Function}}=nothing`: A penalty function for the generator.
- `ϵ::Union{Nothing,AbstractFloat}=nothing`: The tolerance value for the generator. Described at length in Tolomei et al. (https://arxiv.org/pdf/1706.06691.pdf).
- `latent_space::Bool=false`: Whether to use the latent space of the model to generate counterfactuals.

# Returns
- `generator::HeuristicBasedGenerator`: A heuristic-based counterfactual generator.
"""
function HeuristicBasedGenerator(;
    penalty::Union{Nothing,Function,Vector{Function}}=nothing,
    ϵ::Union{Nothing,AbstractFloat}=nothing,
    latent_space::Bool=false,
)
    return HeuristicBasedGenerator(penalty, ϵ, latent_space)
end
