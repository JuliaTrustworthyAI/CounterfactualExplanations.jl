using CounterfactualExplanations: polynomial_decay
using CounterfactualExplanations.Models
using EnergySamplers: EnergySamplers
using LinearAlgebra: LinearAlgebra, det, norm
using Random: Random
using Statistics: mean

abstract type AbstractPenalty end

"""
    distance_mad(ce::AbstractCounterfactualExplanation; agg=mean)

This is the distance measure proposed by Wachter et al. (2017).
"""
function distance_mad(
    ce::AbstractCounterfactualExplanation; agg=Statistics.mean, noise=1e-5, kwrgs...
)
    X = ce.data.X
    mad = []
    ChainRulesCore.ignore_derivatives() do
        _dict = ce.search
        if !(:mad_features ∈ collect(keys(_dict)))
            X̄ = Statistics.median(X; dims=ndims(X))
            _mad = Statistics.median(abs.(X .- X̄); dims=ndims(X))
            _dict[:mad_features] = _mad .+ size(X, 1) * noise        # add noise to avoid division by zero
        end
        _mad = _dict[:mad_features]
        push!(mad, _mad)
    end
    return distance(ce; agg=agg, weights=1.0 ./ mad[1], kwrgs...)
end

"""
    distance_l0(ce::AbstractCounterfactualExplanation)

Computes the L0 distance of the counterfactual to the original factual.
"""
function distance_l0(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=0, kwrgs...)
end

"""
    distance_l1(ce::AbstractCounterfactualExplanation)

Computes the L1 distance of the counterfactual to the original factual.
"""
function distance_l1(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=1, kwrgs...)
end

"""
    distance_l2(ce::AbstractCounterfactualExplanation)

Computes the L2 (Euclidean) distance of the counterfactual to the original factual.
"""
function distance_l2(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=2, kwrgs...)
end

"""
    distance_linf(ce::AbstractCounterfactualExplanation)

Computes the L-inf distance of the counterfactual to the original factual.
"""
function distance_linf(ce::AbstractCounterfactualExplanation; kwrgs...)
    return distance(ce; p=Inf, kwrgs...)
end

"""
    ddp_diversity(
        ce::AbstractCounterfactualExplanation;
        perturbation_size=1e-5
    )

Evaluates how diverse the counterfactuals are using a Determinantal Point Process (DDP).
"""
function ddp_diversity(
    ce::AbstractCounterfactualExplanation; perturbation_size=1e-3, agg=det
)
    X = ce.s′
    xs = eachslice(X; dims=ndims(X))
    K = [1 / (1 + LinearAlgebra.norm(x .- y)) for x in xs, y in xs]
    K += LinearAlgebra.Diagonal(
        Random.randn(eltype(X), size(X)[end]) * convert(eltype(X), perturbation_size)
    )
    cost = -agg(K)
    return cost
end

"""
    distance_from_target(
        ce::AbstractCounterfactualExplanation;
        K::Int=50
    )

Computes the distance of the counterfactual from samples in the target main. If `choose_randomly` is `true`, the function will randomly sample `K` neighbours from the target manifold. Otherwise, it will compute the pairwise distances and select the `K` closest neighbours.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation.
- `K::Int=50`: The number of neighbours to sample.
- `choose_randomly::Bool=true`: Whether to sample neighbours randomly.
- `kwrgs...`: Additional keyword arguments for the distance function.

# Returns
- `Δ::AbstractFloat`: The distance from the counterfactual to the target manifold.
"""
function distance_from_target(
    ce::AbstractCounterfactualExplanation;
    K::Int=50,
    choose_random::Bool=false,
    cosine::Bool=false,
    kwrgs...,
)

    # Get potential neighbours:
    ys = ce.search[:potential_neighbours]
    if K > size(ys, 2)
        @warn "`K` is larger than the number of potential neighbours. Future warnings will be suppressed." maxlog =
            1
    end

    # Get K samples from potential neighbours:
    if choose_random
        # Choose K random samples:
        ids = rand(1:size(ys, 2), K)
    else
        # Get K closest neighbours:
        Δ = []
        ChainRulesCore.ignore_derivatives() do
            δ = map(eachcol(ys)) do y
                distance(ce; from=y, kwrgs...)
            end
            push!(Δ, δ)
        end
        ids = sortperm(Δ[1])[1:K]
    end

    neighbours = ys[:, ids]

    # Compute distance:
    Δ = distance(ce; from=neighbours, cosine=cosine, kwrgs...) / size(neighbours, 2)

    return Δ
end

"""
    function model_loss_penalty(
        ce::AbstractCounterfactualExplanation;
        agg=mean
    )

Additional penalty for ClaPROARGenerator.
"""
function model_loss_penalty(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)
    x_ = CounterfactualExplanations.counterfactual(ce)
    M = ce.M
    model = isa(M.model, LinearAlgebra.Vector) ? M.model : [M.model]
    y_ = ce.target_encoded

    if M.likelihood == :classification_binary
        loss_type = :logitbinarycrossentropy
    else
        loss_type = :logitcrossentropy
    end

    function loss(x, y)
        return sum([getfield(Flux.Losses, loss_type)(nn(x), y) for nn in model]) / length(model)
    end

    return loss(x_, y_)
end

"""
    energy(M::AbstractModel, x::AbstractArray, t::Int)

Computes the energy of the model at a given state as in Altmeyer et al. (2024): https://scholar.google.com/scholar?cluster=3697701546144846732&hl=en&as_sdt=0,5.
"""
function energy(M::AbstractModel, x::AbstractArray, t::Int)
    return -logits(M, x)[t]
end

"""
    energy_constraint(
        ce::AbstractCounterfactualExplanation;
        agg=mean,
        reg_strength::AbstractFloat=0.0,
        decay::AbstractFloat=0.9,
        kwargs...,
    )

Computes the energy constraint for the counterfactual explanation as in Altmeyer et al. (2024): https://scholar.google.com/scholar?cluster=3697701546144846732&hl=en&as_sdt=0,5. The energy constraint is a regularization term that penalizes the energy of the counterfactuals. The energy is computed as the negative logit of the target class.

# Arguments

- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation.
- `agg::Function=mean`: The aggregation function (only applicable in case `num_counterfactuals > 1`). Default is `mean`.
- `reg_strength::AbstractFloat=0.0`: The regularization strength.
- `decay::AbstractFloat=0.9`: The decay rate for the polynomial decay function (defaults to 0.9). Parameter `a` is set to `1.0 / ce.generator.opt.eta`, such that the initial step size is equal to 1.0, not accounting for `b`. Parameter `b` is set to `round(Int, max_steps / 20)`, where `max_steps` is the maximum number of iterations.
- `kwargs...`: Additional keyword arguments.

# Returns

- `ℒ::AbstractFloat`: The energy constraint.
"""
function energy_constraint(
    ce::AbstractCounterfactualExplanation;
    agg=mean,
    reg_strength::AbstractFloat=1e-3,
    decay::AbstractFloat=0.9,
    kwargs...,
)

    # Setup:
    ℒ = 0
    x′ = CounterfactualExplanations.decode_state(ce)     # current state
    t = get_target_index(ce.data.y_levels, ce.target)
    xs = eachslice(x′; dims=ndims(x′))

    # Multiplier ϕ for the energy constraint:
    max_steps = CounterfactualExplanations.Convergence.max_iter(ce.convergence)
    b = round(max_steps / 25)
    a = b / 10
    ϕ = polynomial_decay(a, b, decay, total_steps(ce) + 1)

    # Generative loss:
    gen_loss = energy.(ce.M, xs, t) |> agg

    if reg_strength == 0.0
        ℒ = ϕ * gen_loss
    else
        # Regularization loss:
        reg_loss = norm(energy.(ce.M, xs, t))^2 |> agg

        # Total loss:
        ℒ = ϕ * (gen_loss + reg_strength * reg_loss)
    end

    return ℒ
end

struct EnergyDifferential <: AbstractPenalty
    K::Int
    agg::Function
end

EnergyDifferential(;K::Int=50, agg::Function=mean) = EnergyDifferential(K, agg)
    
function (pen::EnergyDifferential)(ce::AbstractCounterfactualExplanation)

    # If the potential neighbours have not been computed, do so:
    get!(
        ce.search,
        :potential_neighbours,
        CounterfactualExplanations.find_potential_neighbours(ce, pen.K),
    )

    # Get potential neighbours:
    ys = ce.search[:potential_neighbours]
    if pen.K > size(ys, 2)
        @warn "`K` is larger than the number of potential neighbours. Future warnings will be suppressed." maxlog =
            1
    end

    # Get counterfactual:
    x′ = CounterfactualExplanations.decode_state(ce)     # current state
    xs = eachslice(x′; dims=ndims(x′))

    # Compute energy differential:
    Δ = pen.agg(EnergySamplers.energy_differential.(ce.M, xs, (ys,), ce.target))

    return Δ

end

function EnergySamplers.energy_differential(M::AbstractModel, xgen, xsampled, y::Int)
    typeof(M.type) <: Models.AbstractFluxNN || throw(NotImplementedModel(M))
    f = M.fitresult.fitresult
    return EnergySamplers.energy_differential(f, xgen, xsampled, y)
end