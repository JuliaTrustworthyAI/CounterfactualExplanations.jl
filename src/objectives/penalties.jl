using LinearAlgebra: LinearAlgebra, det, norm
using Random: Random
using Statistics: mean

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
    ce::AbstractCounterfactualExplanation; K::Int=50, choose_randomly::Bool=true, kwrgs...
)

    # Get potential neighbours:
    ys = ce.search[:potential_neighbours]
    if K > size(ys, 2)
        @warn "K is larger than the number of potential neighbours. Setting K to the number of potential neighbours."
        K = size(ys, 2)
    end

    # Get K samples from potential neighbours:
    if choose_randomly
        # Choose K random samples:
        ids = rand(1:size(ce.search[:potential_neighbours], 2), K)
    else
        # Compute pairwise distances:

        Δ = map(eachcol(ys)) do y
            distance(ce; from=y, kwrgs...)
        end
        # Get K closest neighbours:
        ids = sortperm(Δ)[1:K]
    end

    neighbours = ys[:, ids]
    centroid = Statistics.mean(neighbours; dims=ndims(neighbours))
    Δ = distance(ce; from=centroid, kwrgs...)
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
        reg_strength=0.1,
        decay::Union{Nothing,Tuple{<:AbstractFloat,<:Int}}=nothing,
        kwargs...,
    )

Computes the energy constraint for the counterfactual explanation as in Altmeyer et al. (2024): https://scholar.google.com/scholar?cluster=3697701546144846732&hl=en&as_sdt=0,5.
"""
function energy_constraint(
    ce::AbstractCounterfactualExplanation;
    agg=mean,
    reg_strength=0.1,
    decay::Union{Nothing,Tuple{<:AbstractFloat,<:Int}}=nothing,
    kwargs...,
)
    ℒ = 0
    x′ = CounterfactualExplanations.decode_state(ce)     # current state

    t = get_target_index(ce.data.y_levels, ce.target)
    xs = eachslice(x′; dims=ndims(x′))

    # Generative loss:
    gen_loss = energy.(ce.M, xs, t) |> agg

    # Regularization loss:
    reg_loss = norm(energy.(ce.M, xs, t))^2 |> agg

    # Decay:
    ϕ = 1.0
    if !isnothing(decay)
        iter = total_steps(ce)
        if iter % decay[2] == 0
            ϕ = exp(-decay[1] * total_steps(ce))
        end
    end

    # Total loss:
    ℒ = ϕ * (gen_loss + reg_strength * reg_loss)

    return ℒ
end
