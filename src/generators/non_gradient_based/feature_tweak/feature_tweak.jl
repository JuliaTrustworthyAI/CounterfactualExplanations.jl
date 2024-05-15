"Feature Tweak counterfactual generator class."
mutable struct FeatureTweakGenerator <: AbstractNonGradientBasedGenerator
    penalty::Union{Nothing,Function,Vector{Function}}
    ϵ::Union{Nothing,AbstractFloat}
    latent_space::Bool
    dim_reduction::Bool
end

"""
    FeatureTweakGenerator(; penalty::Union{Nothing,Function,Vector{Function}}=Objectives.distance_l2, ϵ::AbstractFloat=0.1)

Constructs a new Feature Tweak Generator object.

Uses the L2-norm as the penalty to measure the distance between the counterfactual and the factual.
According to the paper by Tolomei et al., another recommended choice for the penalty in addition to the L2-norm is the L0-norm.
The L0-norm simply minimizes the number of features that are changed through the tweak.

# Arguments
- `penalty::Union{Nothing,Function,Vector{Function}}`: The penalty function to use for the generator. Defaults to `distance_l2`.
- `ϵ::AbstractFloat`: The tolerance value for the feature tweaks. Described at length in Tolomei et al. (https://arxiv.org/pdf/1706.06691.pdf). Defaults to 0.1.

# Returns
- `generator::FeatureTweakGenerator`: A non-gradient-based generator that can be used to generate counterfactuals using the feature tweak method.
"""
function FeatureTweakGenerator(;
    penalty::Union{Nothing,Function,Vector{Function}}=Objectives.distance_l2,
    ϵ::AbstractFloat=0.1,
)
    return FeatureTweakGenerator(penalty, ϵ, false, false)
end
