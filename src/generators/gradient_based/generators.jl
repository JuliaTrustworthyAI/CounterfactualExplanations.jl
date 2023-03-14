"Constructor for `GenericGenerator`."
function GenericGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return Generator(; complexity=Objectives.distance_l2, λ=λ, kwargs...)
end

"Constructor for `DiCEGenerator`."
function DiCEGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, Objectives.ddp_diversity]
    return Generator(; complexity=_penalties, λ=λ, kwargs...)
end

"Constructor for `ClaPGenerator`."
function ClaPROARGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, Objectives.model_loss_penalty]
    return Generator(; complexity=_penalties, λ=λ, kwargs...)
end

"Constructor for `GravitationalGenerator`."
function GravitationalGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 1.0], kwargs...)
    _penalties = [Objectives.distance_l2, Objectives.distance_from_target]
    return Generator(; complexity=_penalties, λ=λ, kwargs...)
end

"Constructor for `REVISEGenerator`."
function REVISEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return Generator(; λ=λ, latent_space=latent_space, kwargs...)
end