"""
    GeneratorConditionsConvergence

Convergence criterion for counterfactual explanations based on the generator conditions. The search stops when the gradients of the search objective are below a certain threshold and the generator conditions are satisfied.

# Fields

- `decision_threshold::AbstractFloat`: The threshold for the decision probability.
- `gradient_tol::AbstractFloat`: The tolerance for the gradients of the search objective.
- `max_iter::Int`: The maximum number of iterations.
- `min_success_rate::AbstractFloat`: The minimum success rate for the generator conditions (across counterfactuals).
"""
struct GeneratorConditionsConvergence <: AbstractConvergence
    decision_threshold::AbstractFloat
    gradient_tol::AbstractFloat
    max_iter::Int
    min_success_rate::AbstractFloat
end

"""
    GeneratorConditionsConvergence(; decision_threshold=0.5, gradient_tol=1e-2, max_iter=100, min_success_rate=0.75, y_levels=nothing)

Outer constructor for `GeneratorConditionsConvergence`.
"""
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
    converged(
        convergence::GeneratorConditionsConvergence,
        ce::AbstractCounterfactualExplanation,
        x::Union{AbstractArray,Nothing}=nothing,
    )

Checks if the counterfactual search has converged when the convergence criterion is generator_conditions.
"""
function converged(
    convergence::GeneratorConditionsConvergence,
    ce::AbstractCounterfactualExplanation,
    x::Union{AbstractArray,Nothing}=nothing,
)
    return threshold_reached(ce, x) && conditions_satisfied(ce.generator, ce)
end

"""
    conditions_satisfied(gen::AbstractGenerator, ce::AbstractCounterfactualExplanation)

This function is overloaded in the `Generators` module to check whether the counterfactual search has converged with respect to generator conditions.
"""
function conditions_satisfied(gen::AbstractGenerator, ce::AbstractCounterfactualExplanation)
    return true
end
