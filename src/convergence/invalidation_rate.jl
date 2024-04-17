using Distributions: Distributions
using Flux: Flux
using LinearAlgebra: LinearAlgebra

Base.@kwdef struct InvalidationRateConvergence <: AbstractConvergence
    invalidation_rate::AbstractFloat = 0.1
    max_iter::Int = 100
    variance::AbstractFloat = 0.01
end

"""
    converged(convergence::InvalidationRateConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is invalidation rate.
"""
function converged(
    convergence::InvalidationRateConvergence, ce::AbstractCounterfactualExplanation
)
    ir = invalidation_rate(ce)
    label = Models.predict_label(ce.M, ce.data, ce.x′)[1]
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
    for i in 1:length(ce.s′)
        push!(
            grad,
            Flux.gradient(
                () -> logits(ce.M, CounterfactualExplanations.decode_state(ce))[i],
                Flux.params(ce.s′),
            )[ce.s′],
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

"""
    hinge_loss(convergence::InvalidationRateConvergence, ce::AbstractCounterfactualExplanation)

Calculates the hinge loss of a counterfactual explanation.

# Arguments
- `convergence::InvalidationRateConvergence`: The convergence criterion to use.
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the hinge loss for.

# Returns
The hinge loss of the counterfactual explanation.
"""
function hinge_loss(
    convergence::InvalidationRateConvergence, ce::AbstractCounterfactualExplanation
)
    return max(0, invalidation_rate(ce) - convergence.invalidation_rate)
end
