"""
    _replace_nans(grad_ce_state::AbstractArray, old_new::Pair=(NaN => 0))

Helper function to deal with exploding gradients. This is only a temporary fix and will be improved.
"""
function _replace_nans(grad_ce_state::AbstractArray, old_new::Pair=(NaN => 0))
    return replace(grad_ce_state, old_new)
end

"""
    Convergence.conditions_satisfied(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
By default, gradient-based search is considered to have converged as soon as the proposed feature changes for all features are smaller than one percent of its standard deviation.
"""
function Convergence.conditions_satisfied(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    grad_ce_state = grad_search_opt(generator, ce)
    grad_ce_state = CounterfactualExplanations.apply_mutability(ce, grad_ce_state)
    if !hasfield(typeof(ce.convergence), :gradient_tol)
        # Temporary fix due to the fact that `ProbeGenerator` relies on `InvalidationRateConvergence`.
        @warn "Checking for generator conditions convergence is not implemented for this generator type. Return `false`." maxlog =
            1
        return false
    end
    τ = ce.convergence.gradient_tol
    satisfied = map(
        x -> all(abs.(x) .< τ), eachslice(grad_ce_state; dims=ndims(grad_ce_state))
    )
    success_rate = sum(satisfied) / ce.num_counterfactuals
    status = success_rate > ce.convergence.min_success_rate
    return status
end
