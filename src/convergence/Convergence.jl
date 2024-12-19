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

function converged(
    convergence::Symbol,
    ce::AbstractCounterfactualExplanation,
    y_levels::AbstractVector,
    x::Union{AbstractArray,Nothing}=nothing,
)
    conv = get_convergence_type(convergence, y_levels)
    return converged(conv, ce, x)
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
