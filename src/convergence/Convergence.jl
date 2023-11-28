module Convergence

using .Generators
using .Models

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
    :early_stopping => EarlyStoppingConvergence(),
)

export convergence_catalogue
export converged
export DecisionThresholdConvergence,
    GeneratorConditionsConvergence, InvalidationRateConvergence, MaxIterConvergence

end
