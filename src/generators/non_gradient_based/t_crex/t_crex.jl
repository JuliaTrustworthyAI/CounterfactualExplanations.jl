using CounterfactualExplanations
using StatsBase

"T-CREx counterfactual generator class."
mutable struct TCRExGenerator <: AbstractNonGradientBasedGenerator 
    ρ::AbstractFloat
    τ::AbstractFloat
    forest::Bool
end

function TCRExGenerator(; ρ::AbstractFloat=0.2, τ::AbstractFloat=0.9, forest::Bool=false)
    return TCRExGenerator(ρ, τ, forest)
end