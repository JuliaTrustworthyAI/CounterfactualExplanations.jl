"""
    _replace_nans(Δs′::AbstractArray, old_new::Pair=(NaN => 0))

Helper function to deal with exploding gradients. This is only a temporary fix and will be improved.
"""
function _replace_nans(Δs′::AbstractArray, old_new::Pair=(NaN => 0))
    return replace(Δs′, old_new)
end

"""
    conditions_satisfied(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
By default, gradient-based search is considered to have converged as soon as the proposed feature changes for all features are smaller than one percent of its standard deviation.
"""
function conditions_satisfied(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    Δs′ = ∇(generator, ce.M, ce)
    Δs′ = CounterfactualExplanations.apply_mutability(ce, Δs′)
    τ = ce.convergence.gradient_tol
    satisfied = map(x -> all(abs.(x) .< τ), eachslice(Δs′; dims=ndims(Δs′)))
    success_rate = sum(satisfied) / ce.num_counterfactuals
    status = success_rate > ce.convergence.min_success_rate
    return status
end
