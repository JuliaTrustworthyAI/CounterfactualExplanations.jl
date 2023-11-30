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
    λ::AbstractFloat=0.1,
    loss::Symbol=:logitbinarycrossentropy,
    penalty::Penalty=Objectives.distance_l1,
    kwargs...,
)
    @assert haskey(losses_catalogue, loss) "Loss function not found in catalogue."
    user_loss = Objectives.losses_catalogue[loss]
    return GradientBasedGenerator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
end
