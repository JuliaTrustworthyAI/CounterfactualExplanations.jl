Base.@kwdef struct GeneratorConditionsConvergence <: AbstractConvergence
    decision_threshold::AbstractFloat = 0.5
    gradient_tol::AbstractFloat = 1e-2
    max_iter::Int = 100
    min_success_rate::AbstractFloat = 0.75
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