import ..Objectives: losses_catalogue, distance_l1
import LinearAlgebra: transpose, Matrix
import Flux.Losses
import Distributions: cdf, Normal
"""
    ProbeGenerator(; λ::AbstractFloat=0.1, loss::Symbol=:mse, penalty=distance_l1, kwargs...)

Create a generator that generates counterfactual probes using the specified loss function and penalty function.

# Arguments
- `λ::AbstractFloat`: The regularization parameter for the generator.
- `loss::Symbol`: The loss function to use for the generator. Defaults to `:mse`.
- `penalty`: The penalty function to use for the generator. Defaults to `distance_l1`.
- `kwargs`: Additional keyword arguments to pass to the `Generator` constructor.

# Returns
A `Generator` object that can be used to generate counterfactual probes.

based on https://arxiv.org/abs/2203.06768
"""
function ProbeGenerator(;
    λ::AbstractFloat=0.1,
    loss::Symbol=:logitbinarycrossentropy,
    penalty=distance_l1,
    kwargs...,
)
    @assert haskey(losses_catalogue, loss) "Loss function not found in catalogue."
    user_loss = losses_catalogue[loss]
    return Generator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
end

"""
    invalidation_rate(ce::AbstractCounterfactualExplanation)

Calculate the invalidation rate of a counterfactual explanation.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the invalidation rate for.
- `kwargs`: Additional keyword arguments to pass to the function.

# Returns
The invalidation rate of the counterfactual explanation.
"""
function invalidation_rate(ce::AbstractCounterfactualExplanation)
    f_loss = logits(ce.M, CounterfactualExplanations.decode_state(ce))[ce.target]
    grad = []
    # This has to be done with a for loop because flux does not know how to take a gradient from an array of logits.
    for i in 1:length(ce.s′)
        push!(
            grad,
            gradient(
                () -> logits(ce.M, CounterfactualExplanations.decode_state(ce))[i],
                Flux.params(ce.s′),
            )[ce.s′],
        )
    end
    gradᵀ = transpose(grad)

    identity_matrix = Matrix{Float32}(I, length(grad), length(grad))
    denominator = sqrt(gradᵀ * ce.params[:variance] * identity_matrix * grad)[1]

    normalized_gradient = f_loss / denominator
    ϕ = cdf(Normal(0, 1), normalized_gradient)
    return 1 - ϕ
end

"""
    hinge_loss(ce::AbstractCounterfactualExplanation)

Calculate the hinge loss of a counterfactual explanation.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the hinge loss for.

# Returns
The hinge loss of the counterfactual explanation.
"""
function hinge_loss(ce::AbstractCounterfactualExplanation)
    return max(0, invalidation_rate(ce) - ce.params[:invalidation_rate])
end
