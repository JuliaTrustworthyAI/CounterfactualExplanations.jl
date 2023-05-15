function FeatureTweakGenerator(; ϵ::AbstractFloat, kwargs...)
    return HeuristicBasedGenerator(; penalty=Objectives.distance_l2, ϵ=ϵ, kwargs...)
end