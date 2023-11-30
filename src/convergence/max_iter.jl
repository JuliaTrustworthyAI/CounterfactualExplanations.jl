Base.@kwdef struct MaxIterConvergence <: AbstractConvergence
    max_iter::Int = 100
end

"""
    converged(convergence::MaxIterConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is maximum iterations.
"""
function converged(convergence::MaxIterConvergence, ce::AbstractCounterfactualExplanation)
    return ce.search[:iteration_count] == ce.convergence.max_iter
end
