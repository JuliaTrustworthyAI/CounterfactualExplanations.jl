"Base class for `DecisionThresholdConvergence`."
mutable struct DecisionThresholdConvergence <: AbstractConvergenceType
    max_iter::Int
    decision_threshold::AbstractFloat
    min_success_rate::AbstractFloat
end

"Base class for `GeneratorConditionsConvergence`."
mutable struct GeneratorConditionsConvergence <: AbstractConvergenceType
    max_iter::Int
    min_success_rate::AbstractFloat
    gradient_tol::AbstractFloat
end

"Base class for `MaxIterConvergence`."
mutable struct MaxIterConvergence <: AbstractConvergenceType
    max_iter::Int
end

"Base class for `InvalidationRateConvergence`."
mutable struct InvalidationRateConvergence <: AbstractConvergenceType
    max_iter::Int
    invalidation_rate::AbstractFloat
    variance::AbstractFloat
end

"Base class for `EarlyStoppingConvergence`."
mutable struct EarlyStoppingConvergence <: AbstractConvergenceType
    max_iter::Int
end


"""
Outer constructor for `DecisionThresholdConvergence`.
"""
function DecisionThresholdConvergence(; max_iter=100, decision_threshold=0.5, min_success_rate=parameters[:min_success_rate])
    @assert 0.0 < min_success_rate <= 1.0 "Minimum success rate should be ∈ [0.0,1.0]."
    return DecisionThresholdConvergence(max_iter, decision_threshold, min_success_rate)
end

"""
Outer constructor for `GeneratorConditionsConvergence`.
"""
function GeneratorConditionsConvergence(; max_iter=100, min_success_rate=parameters[:min_success_rate], gradient_tol=parameters[:τ])
    @assert 0.0 < min_success_rate <= 1.0 "Minimum success rate should be ∈ [0.0,1.0]."
    return GeneratorConditionsConvergence(max_iter, min_success_rate, gradient_tol)
end

"""
Outer constructor for `MaxIterConvergence`.
"""
function MaxIterConvergence(; max_iter=100)
    return MaxIterConvergence(max_iter)
end

"""
Outer constructor for `InvalidationRateConvergence`.
"""
function InvalidationRateConvergence(; max_iter=100, invalidation_rate=0.1, variance=0.01)
    return InvalidationRateConvergence(max_iter, invalidation_rate, variance)
end

"""
Outer constructor for `EarlyStoppingConvergence`.
"""
function EarlyStoppingConvergence(; max_iter=100)
    return EarlyStoppingConvergence(max_iter)
end
