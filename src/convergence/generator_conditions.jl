Base.@kwdef struct GeneratorConditionsConvergence <: AbstractConvergence
    max_iter::Int = 100
    min_success_rate::AbstractFloat = 0.75
    gradient_tol::AbstractFloat = 1e-2
end

"""
    converged(convergence::GeneratorConditionsConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is generator_conditions.
"""
function converged(
    convergence::GeneratorConditionsConvergence, ce::CounterfactualExplanation
)
    return threshold_reached(ce) && Generators.conditions_satisfied(ce.generator, ce)
end
