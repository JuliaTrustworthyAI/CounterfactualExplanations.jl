const default_distance = Objectives.distance_l1

"Constructor for `GenericGenerator`."
function GenericGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=default_distance, λ=λ, kwargs...)
end

const DOC_ECCCo = "For details, see Altmeyer et al. ([2024](https://ojs.aaai.org/index.php/AAAI/article/view/28956))."

"Constructor for `ECCoGenerator`. This corresponds to the generator proposed in https://arxiv.org/abs/2312.10648, without the conformal set size penalty. $DOC_ECCCo"
function ECCoGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 1.0], kwargs...)
    _penalties = [default_distance, Objectives.energy_constraint]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

const DOC_Wachter = "For details, see Wachter et al. ([2018](https://arxiv.org/abs/1711.00399))."

"Constructor for `WachterGenerator`. $DOC_Wachter"
function WachterGenerator(; λ::AbstractFloat=0.1, kwargs...)
    return GradientBasedGenerator(; penalty=Objectives.distance_mad, λ=λ, kwargs...)
end

const DOC_DiCE = "For details, see Mothilal et al. ([2020](https://arxiv.org/abs/1905.07697))."

"Constructor for `DiCEGenerator`. $DOC_DiCE"
function DiCEGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.1], kwargs...)
    _penalties = [default_distance, Objectives.ddp_diversity]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

const DOC_SaTML = "For details, see Altmeyer et al. ([2023](https://ieeexplore.ieee.org/abstract/document/10136130))."

"Constructor for `ClaPGenerator`. $DOC_SaTML"
function ClaPROARGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.model_loss_penalty]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

"Constructor for `GravitationalGenerator`. $DOC_SaTML"
function GravitationalGenerator(; λ::Vector{<:AbstractFloat}=[0.1, 0.5], kwargs...)
    _penalties = [default_distance, Objectives.distance_from_target]
    return GradientBasedGenerator(; penalty=_penalties, λ=λ, kwargs...)
end

const DOC_REVISE = "For details, see Joshi et al. ([2019](https://arxiv.org/abs/1907.09615))."

"Constructor for `REVISEGenerator`. $DOC_REVISE"
function REVISEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        penalty=default_distance, λ=λ, latent_space=latent_space, kwargs...
    )
end

const DOC_Greedy = "For details, see Schut et al. ([2021](https://proceedings.mlr.press/v130/schut21a/schut21a.pdf))."

"Constructor for `GreedyGenerator`. $DOC_Greedy"
function GreedyGenerator(; η=0.1, n=nothing, kwargs...)
    opt = CounterfactualExplanations.Generators.JSMADescent(; η=η, n=n)
    return GradientBasedGenerator(; penalty=default_distance, λ=0.0, opt=opt, kwargs...)
end

const DOC_CLUE = "For details, see Antoran et al. ([2021](https://arxiv.org/abs/2006.06848))."

"Constructor for `CLUEGenerator`. $DOC_CLUE"
function CLUEGenerator(; λ::AbstractFloat=0.1, latent_space=true, kwargs...)
    return GradientBasedGenerator(;
        loss=predictive_entropy,
        penalty=default_distance,
        λ=λ,
        latent_space=latent_space,
        kwargs...,
    )
end

const DOC_Probe = "For details, see Pawelczyk et al. ([2022](https://proceedings.mlr.press/v151/pawelczyk22a/pawelczyk22a.pdf))."

const DOC_Probe_warn = "The `ProbeGenerator` is currenlty not working adequately. In particular, gradients are not computed with respect to the Hinge loss term proposed in the paper. It is still possible, however, to use this generator to achieve a desired invalidation rate. See issue [#376](https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl/issues/376) for details."

"""
Constructor for `ProbeGenerator`. $DOC_Probe

## Warning

$DOC_Probe_warn
"""
function ProbeGenerator(;
    λ::Vector{<:AbstractFloat}=[0.1, 1.0],
    loss::Symbol=:logitbinarycrossentropy,
    penalty=[Objectives.distance_l1, Objectives.hinge_loss],
    kwargs...,
)
    @warn DOC_Probe_warn
    user_loss = Objectives.losses_catalogue[loss]
    return GradientBasedGenerator(; loss=user_loss, penalty=penalty, λ=λ, kwargs...)
end
