"""
    FeatureTweakGenerator(; ϵ::AbstractFloat, kwargs...)

Constructs a new Feature Tweak Generator object.

Uses the L2-norm as the penalty to measure the distance between the counterfactual and the factual.
According to the paper by Tolomei er al., an alternative choice here would be using the L0-norm to simply minimize the number of features that are changed through the tweak.

# Arguments
- `ϵ::AbstractFloat`: The tolerance value for the feature tweaks. Described at length in Tolomei et al. (https://arxiv.org/pdf/1706.06691.pdf).

# Returns
- `generator::HeuristicBasedGenerator`: A non-gradient-based generator that can be used to generate counterfactuals using the feature tweak method.
"""
function FeatureTweakGenerator(; ϵ::AbstractFloat=0.1, kwargs...)
    return HeuristicBasedGenerator(; penalty=Objectives.distance_l2, ϵ=ϵ, kwargs...)
end
