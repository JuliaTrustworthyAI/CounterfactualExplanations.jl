"""
    ProbeGenerator(;
        λ::AbstractFloat=0.1,
        loss::Symbol=:logitbinarycrossentropy,
        penalty::Penalty=Objectives.distance_l1,
        kwargs...,
    )

Create a generator that generates counterfactual probes using the specified loss function and penalty function.

# Arguments
- `λ::AbstractFloat`: The regularization parameter for the generator.
- `loss::Symbol`: The loss function to use for the generator. Defaults to `:mse`.
- `penalty::Penalty`: The penalty function to use for the generator. Defaults to `distance_l1`.
- `kwargs`: Additional keyword arguments to pass to the `Generator` constructor.

# Returns
A `Generator` object that can be used to generate counterfactual probes.

based on https://arxiv.org/abs/2203.06768
"""
function ProbeGenerator(;
    λ::Vector{<:AbstractFloat}=[1.0, 0.1],
    loss::Symbol=:logitbinarycrossentropy,
    penalty::Penalty=[hinge_loss_ir, Objectives.distance_l1],
    kwargs...,
)
    @assert haskey(losses_catalogue, loss) "Loss function not found in catalogue."
    user_loss = Objectives.losses_catalogue[loss]
    return GradientBasedGenerator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
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
    if !hasfield(typeof(ce.convergence), :invalidation_rate)
        @warn "Invalidation rate is only defined for InvalidationRateConvergence. Returning 0."
        return 0.0
    end

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

    identity_matrix = LinearAlgebra.Matrix{Float32}(I, length(grad), length(grad))
    denominator = sqrt(gradᵀ * ce.convergence.variance * identity_matrix * grad)[1]

    normalized_gradient = f_loss / denominator
    ϕ = Distributions.cdf(Distributions.Normal(0, 1), normalized_gradient)
    return 1 - ϕ
end

"""
    hinge_loss_ir(convergence::InvalidationRateConvergence, ce::AbstractCounterfactualExplanation)

Calculate the hinge loss of a counterfactual explanation with respect to the probability of invalidation following: https://openreview.net/forum?id=sC-PmTsiTB.

# Arguments
- `convergence::InvalidationRateConvergence`: The convergence criterion to use.
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the hinge loss for.

# Returns
The hinge loss of the counterfactual explanation.
"""
function hinge_loss_ir(ce::AbstractCounterfactualExplanation)
    if !hasfield(typeof(ce.convergence), :invalidation_rate)
        @warn "Invalidation rate is only defined for InvalidationRateConvergence. Returning 0."
        return 0.0
    end
    return max(0, invalidation_rate(ce) - ce.convergence.invalidation_rate)
end

# Add the hinge loss to the losses catalogue.
Objectives.losses_catalogue[:hinge_loss_ir] = hinge_loss_ir
