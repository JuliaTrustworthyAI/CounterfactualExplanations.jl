"""
Uses the L2-norm as the penalty to measure the distance between the counterfactual and the factual.
According to the paper by Tolomei er al., an alternative choice here would be using the L0-norm to simply minimize the number of features that are changed through the tweak.
"""
function FeatureTweakGenerator(; ϵ::AbstractFloat=0.1, kwargs...)
    return HeuristicBasedGenerator(; penalty=Objectives.distance_l2, ϵ=ϵ, kwargs...)
end
