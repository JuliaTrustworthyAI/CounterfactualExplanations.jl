const default_distance = Objectives.distance_l1

"Constructor for `GenericGenerator`."
function GenericGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=default_distance, λ=λ, kwargs...)
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
function ClaPROARGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.1], kwargs...)
    _penalties = [default_distance, Objectives.model_loss_penalty]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `GravitationalGenerator`."
function GravitationalGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.1], kwargs...)
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
