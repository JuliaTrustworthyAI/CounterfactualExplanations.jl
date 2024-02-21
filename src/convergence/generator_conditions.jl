struct GeneratorConditionsConvergence <: AbstractConvergence
    decision_threshold::AbstractFloat
    gradient_tol::AbstractFloat
    max_iter::Int
    min_success_rate::AbstractFloat
end

function GeneratorConditionsConvergence(;
    decision_threshold::AbstractFloat=0.5,
    gradient_tol::AbstractFloat=1e-2,
    max_iter::Int=100,
    min_success_rate::AbstractFloat=0.75,
    y_levels::Union{Nothing,AbstractVector}=nothing,
)
    @assert 0.0 < min_success_rate <= 1.0 "Minimum success rate should be ∈ [0.0,1.0]."
    if isa(y_levels, AbstractVector)
        decision_threshold = 1 / length(y_levels)
    end
    return GeneratorConditionsConvergence(
        decision_threshold, gradient_tol, max_iter, min_success_rate
    )
end

"""
    converged(convergence::GeneratorConditionsConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is generator_conditions.
"""
function converged(
    convergence::GeneratorConditionsConvergence, ce::AbstractCounterfactualExplanation
)
    return threshold_reached(ce) && Generators.conditions_satisfied(ce.generator, ce)
end
