"""
    _replace_nans(Δcounterfactual_state::AbstractArray, old_new::Pair=(NaN => 0))

Helper function to deal with exploding gradients. This is only a temporary fix and will be improved.
"""
function _replace_nans(Δcounterfactual_state::AbstractArray, old_new::Pair=(NaN => 0))
    return replace(Δcounterfactual_state, old_new)
end

"""
    Convergence.conditions_satisfied(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
By default, gradient-based search is considered to have converged as soon as the proposed feature changes for all features are smaller than one percent of its standard deviation.
"""
function Convergence.conditions_satisfied(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    if !hasfield(ce.convergence, :gradient_tol)
        # Temporary fix due to the fact that `ProbeGenerator` relies on `InvalidationRateConvergence`.
        @warn "Checking for generator conditions convergence is not implemented for this generator type. Return `false`." maxlog =
            1
        return false
    end
    Δcounterfactual_state = ∇(generator, ce)
    Δcounterfactual_state = CounterfactualExplanations.apply_mutability(
        ce, Δcounterfactual_state
    )
    τ = ce.convergence.gradient_tol
    satisfied = map(
        x -> all(abs.(x) .< τ),
        eachslice(Δcounterfactual_state; dims=ndims(Δcounterfactual_state)),
    )
    success_rate = sum(satisfied) / ce.num_counterfactuals
    status = success_rate > ce.convergence.min_success_rate
    return status
end
