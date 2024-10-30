using Distributions: Distributions
using Flux: Flux
using LinearAlgebra: LinearAlgebra

Base.@kwdef struct InvalidationRateConvergence <: AbstractConvergence
    invalidation_rate::AbstractFloat = 0.1
    max_iter::Int = 100
    variance::AbstractFloat = 0.01
end

"""
    converged(
        convergence::InvalidationRateConvergence,
        ce::AbstractCounterfactualExplanation,
        x::Union{AbstractArray,Nothing}=nothing,
    )

Checks if the counterfactual search has converged when the convergence criterion is invalidation rate.
"""
function converged(
    convergence::InvalidationRateConvergence,
    ce::AbstractCounterfactualExplanation,
    x::Union{AbstractArray,Nothing}=nothing,
)
    ir = invalidation_rate(ce)
    label = Models.predict_label(ce.M, ce.data, ce.counterfactual)[1]
    return label == ce.target && convergence.invalidation_rate > ir
end

"""
    invalidation_rate(ce::AbstractCounterfactualExplanation)

Calculates the invalidation rate of a counterfactual explanation.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the invalidation rate for.
- `kwargs`: Additional keyword arguments to pass to the function.

# Returns
The invalidation rate of the counterfactual explanation.
"""
function invalidation_rate(ce::AbstractCounterfactualExplanation)
    index_target = findfirst(map(x -> x == ce.target, ce.data.y_levels))
    f_loss = logits(ce.M, CounterfactualExplanations.decode_state(ce))[index_target]
    grad = []
    for i in 1:length(ce.counterfactual_state)
        push!(
            grad,
            Flux.gradient(
                () -> logits(ce.M, CounterfactualExplanations.decode_state(ce))[i],
                Flux.params(ce.counterfactual_state),
            )[ce.counterfactual_state],
        )
    end
    gradᵀ = LinearAlgebra.transpose(grad)

    identity_matrix = LinearAlgebra.Matrix{Float32}(
        LinearAlgebra.I, length(grad), length(grad)
    )
    denominator = sqrt(gradᵀ * ce.convergence.variance * identity_matrix * grad)[1]

    normalized_gradient = f_loss / denominator
    ϕ = Distributions.cdf(Distributions.Normal(0, 1), normalized_gradient)
    return 1 - ϕ
end
