import ..Objectives: losses_catalogue, distance_l1
import LinearAlgebra: transpose, Matrix
import Flux.Losses
import Distributions: cdf, Normal

function ProbeGenerator(;
    λ::AbstractFloat=0.1, loss::Symbol=:mse, penalty=distance_l1, kwargs...
)
    @assert haskey(losses_catalogue, loss) "Loss function not found in catalogue."
    user_loss = losses_catalogue[loss]
    # combined_loss = combine_losses(user_loss)
    return Generator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
end
export ProbeGenerator

function combine_losses(user_loss::Function)
    return ce::AbstractCounterfactualExplanation -> user_loss(ce) + hingeLoss(ce)
end

# function hingeLoss(ce::AbstractCounterfactualExplanation; σ²=0.01, kwargs...)
#     f_loss = Flux.Losses.mse(
#         logits(ce.M, CounterfactualExplanations.decode_state(ce)),
#         ce.target_encoded;
#         kwargs...,
#     )
#     # grad = ∂ℓ(ce.generator, ce.M, ce)
#     grad = gradient(
#         () -> Flux.Losses.mse(
#             logits(ce.M, CounterfactualExplanations.decode_state(ce)),
#             ce.target_encoded;
#             kwargs...,
#         ),
#         Flux.params(ce.s′),
#     )[ce.s′]
#     gradᵀ = transpose(grad)

#     identity_matrix = Matrix{Float64}(I, length(grad), length(grad))
#     denominator = sqrt(gradᵀ * σ² * identity_matrix * grad)[1]

#     normalized_gradient = f_loss / denominator
#     ϕ = cdf(Normal(0, 1), normalized_gradient)
#     println(normalized_gradient, "< norm > fi ", ϕ)
#     return max(0, 1 - ϕ)
# end

function hingeLoss(ce::AbstractCounterfactualExplanation; σ²=0.01, kwargs...)
    # println("ce.target_encoded: ", ce.target)
    f_loss = logits(ce.M, CounterfactualExplanations.decode_state(ce))[ce.target]
    # println("f_loss: ", f_loss)
    println(CounterfactualExplanations.decode_state(ce))
    grad = gradient(
        () -> logits(ce.M, CounterfactualExplanations.decode_state(ce))[ce.target],
        Flux.params(ce.s′),
    )[ce.s′]
    # f_loss = Flux.Losses.mse(
    #     logits(ce.M, CounterfactualExplanations.decode_state(ce)),
    #     ce.target_encoded;
    #     kwargs...,
    # )
    # # grad = ∂ℓ(ce.generator, ce.M, ce)
    # grad = gradient(
    #     () -> Flux.Losses.mse(
    #         logits(ce.M, CounterfactualExplanations.decode_state(ce)),
    #         ce.target_encoded;
    #         kwargs...,
    #     ),
    #     Flux.params(ce.s′),
    # )[ce.s′]
    gradᵀ = transpose(grad)

    identity_matrix = Matrix{Float64}(I, length(grad), length(grad))
    denominator = sqrt(gradᵀ * σ² * identity_matrix * grad)[1]

    normalized_gradient = f_loss / denominator
    ϕ = cdf(Normal(0, 1), normalized_gradient)
    # println(normalized_gradient, "< norm > fi ", ϕ)
    return max(0, 1 - ϕ)
end

function R_loss_function(ce::AbstractCounterfactualExplanation)
    return max(0, hingeLoss(ce) - 0.5)
end