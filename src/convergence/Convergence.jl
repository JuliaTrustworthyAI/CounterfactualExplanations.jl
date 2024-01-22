module Convergence

using Distributions
using Flux
using LinearAlgebra
using ..CounterfactualExplanations
using ..Generators
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
    :decision_threshold => DecisionThresholdConvergence(),
    :generator_conditions => GeneratorConditionsConvergence(),
    :max_iter => MaxIterConvergence(),
    :invalidation_rate => InvalidationRateConvergence(),
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
        convergence(y_levels),
        () -> error("Convergence criterion not recognized: $convergence."),
    )
end

export convergence_catalogue
export converged
export get_convergence_type
export hinge_loss, invalidation_rate
export threshold_reached
export DecisionThresholdConvergence
export GeneratorConditionsConvergence
export InvalidationRateConvergence
export MaxIterConvergence

end
