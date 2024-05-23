const default_distance = Objectives.distance_l1

"Constructor for `GenericGenerator`."
function GenericGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=default_distance, λ=λ, kwargs...)
end

"Constructor for `ECCoGenerator`. This corresponds to the generator proposed in https://arxiv.org/abs/2312.10648, without the conformal set size penalty."
function ECCoGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.energy_constraint]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `WachterGenerator`."
function WachterGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=Objectives.distance_mad, λ=λ, kwargs...)
end

"Constructor for `DiCEGenerator`."
function DiCEGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.1], kwargs...)
    _penalties = [default_distance, Objectives.ddp_diversity]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `ClaPGenerator`."
function ClaPROARGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.model_loss_penalty]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `GravitationalGenerator`."
function GravitationalGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.distance_from_target]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `REVISEGenerator`."
function REVISEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        penalty=default_distance, λ=λ, latent_space=latent_space, kwargs...
    )
end

"Constructor for `GreedyGenerator`."
function GreedyGenerator(; η=0.1, n=nothing, kwargs...)
    opt = CounterfactualExplanations.Generators.JSMADescent(; η=η, n=n)
    return GradientBasedGenerator(; penalty=default_distance, λ=0.0, opt=opt, kwargs...)
end

"Constructor for `CLUEGenerator`."
function CLUEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        loss=predictive_entropy,
        penalty=default_distance,
        λ=λ,
        latent_space=latent_space,
        kwargs...,
    )
end

"Constructor for `ProbeGenerator`."
function ProbeGenerator(;
    λ::AbstractFloat=0.1,
    loss::Symbol=:logitbinarycrossentropy,
    penalty=Objectives.distance_l1,
    kwargs...,
)
    @assert haskey(losses_catalogue, loss) "Loss function not found in catalogue."
    user_loss = Objectives.losses_catalogue[loss]
    return GradientBasedGenerator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
end
