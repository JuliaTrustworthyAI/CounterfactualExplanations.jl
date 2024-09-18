"T-CREx counterfactual generator class."
mutable struct TCRExGenerator <: AbstractNonGradientBasedGenerator 
    ρ::AbstractFloat
    τ::AbstractFloat
    forest::Bool
end

function TCRExGenerator(ρ::AbstractFloat=0.2, τ::AbstractFloat=0.9; forest::Bool=false)
    return TCRExGenerator(ρ, τ, forest)
end

function grow_surrogate end

function extract_rules end

function rule_acc(rule, ce::CounterfactualExplanation)
    X = ce.data.X
    checks = 0
    for x in eachcol(X)
        for (lb,ub) in rule
            checks += lb <= x < ub 
        end
    end
    return checks / size(X, 2)
end