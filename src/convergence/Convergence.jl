module Convergence

using Distributions
using Flux: Flux
using LinearAlgebra
using ..CounterfactualExplanations
using ..Models

include("decision_threshold.jl")
include("generator_conditions.jl")
include("invalidation_rate.jl")
include("max_iter.jl")

"""
    converged(ce::AbstractCounterfactualExplanation)

Returns `true` if the counterfactual explanation has converged.
"""
function converged(ce::AbstractCounterfactualExplanation)
    _conv = converged(ce.convergence, ce)
    ce.search[:converged] = _conv
    return _conv
end

"""
    convergence_catalogue

A dictionary containing all convergence criteria.
"""
const convergence_catalogue = Dict(
    :decision_threshold => (y_levels) -> DecisionThresholdConvergence(; y_levels),
    :generator_conditions => (y_levels) -> GeneratorConditionsConvergence(; y_levels),
    :max_iter => (y_levels) -> MaxIterConvergence(),
    :invalidation_rate => (y_levels) -> InvalidationRateConvergence(),
)

"""
    get_convergence_type(convergence::AbstractConvergence)

Returns the convergence object.
"""
function get_convergence_type(convergence::AbstractConvergence, y_levels::AbstractVector)
    return convergence
end

"""
    get_convergence_type(convergence::Symbol)

Returns the convergence object from the dictionary of default convergence types.
"""
function get_convergence_type(convergence::Symbol, y_levels::AbstractVector)
    return get(
        convergence_catalogue,
        convergence,
        () -> error("Convergence criterion not recognized: $convergence."),
    )(
        y_levels
    )
end

"""
    max_iter(conv::AbstractConvergence)

Returns the maximum number of iterations specified.
"""
function max_iter(conv::AbstractConvergence)
    return conv.max_iter
end

export convergence_catalogue
export converged
export get_convergence_type
export invalidation_rate
export threshold_reached
export DecisionThresholdConvergence
export GeneratorConditionsConvergence
export InvalidationRateConvergence
export MaxIterConvergence
export conditions_satisfied

end
