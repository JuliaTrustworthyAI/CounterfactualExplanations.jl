Base.@kwdef struct MaxIterConvergence <: AbstractConvergence
    max_iter::Int = 100
end

"""
    converged(convergence::MaxIterConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is early stopping.
"""
function converged(convergence::MaxIterConvergence, ce::CounterfactualExplanation)
    return ce.search[:iteration_count] == ce.convergence.max_iter
end
