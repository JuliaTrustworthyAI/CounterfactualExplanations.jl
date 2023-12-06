Base.@kwdef struct MaxIterConvergence <: AbstractConvergence
    max_iter::Int = 100
end

"""
    converged(convergence::MaxIterConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is maximum iterations. This means the counterfactual search will not terminate until the maximum number of iterations has been reached independently of the other convergence criteria.
"""
function converged(convergence::MaxIterConvergence, ce::AbstractCounterfactualExplanation)
    return ce.search[:iteration_count] == convergence.max_iter
end
