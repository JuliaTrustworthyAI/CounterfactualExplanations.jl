using ChainRulesCore
using ..CounterfactualExplanations
using LinearAlgebra
using SliceMap
using Statistics: mean, median

"""
    distance(ce::AbstractCounterfactualExplanation, p::Real=2)

Computes the distance of the counterfactual to the original factual.
"""
function distance(
    ce::AbstractCounterfactualExplanation;
    from::Union{Nothing,AbstractArray}=nothing,
    agg=mean,
    p::Real=1,
    weights::Union{Nothing,AbstractArray}=nothing,
)
    if isnothing(from)
        from = CounterfactualExplanations.factual(ce)
    end
    x = CounterfactualExplanations.factual(ce)
    x′ = CounterfactualExplanations.counterfactual(ce)
    xs = eachslice(x′; dims=ndims(x′))                      # slices along the last dimension (i.e. the number of counterfactuals)
    if isnothing(weights)
        Δ = agg(map(x′ -> norm(x′ .- from, p), xs))            # aggregate across counterfactuals
    else
        @assert length(weights) == size(first(xs), ndims(first(xs))) "The length of the weights vector must match the number of features."
        Δ = agg(map(x′ -> (norm.(x′ .- from, p)'weights)[1], xs))   # aggregate across counterfactuals
    end
    return Δ
end

"""
    distance_mad(ce::AbstractCounterfactualExplanation; agg=mean)

This is the distance measure proposed by Wachter et al. (2017).
"""
function distance_mad(ce::AbstractCounterfactualExplanation; agg=mean, noise=1e-5, kwrgs...)
    X = ce.data.X
    mad = []
    ignore_derivatives() do
        _dict = ce.params
        if !(:mad_features ∈ collect(keys(_dict)))
            X̄ = median(X; dims=ndims(X))
            _mad = median(abs.(X .- X̄); dims=ndims(X))
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
    K = [1 / (1 + norm(x .- y)) for x in xs, y in xs]
    K += LinearAlgebra.Diagonal(
        randn(eltype(X), size(X)[end]) * convert(eltype(X), perturbation_size)
    )
    cost = -agg(K)
    return cost
end

"""
    distance_from_target(
        ce::AbstractCounterfactualExplanation, p::Int=2; 
        agg=mean, K::Int=50
    )

Computes the distance of the counterfactual from a point in the target main.
"""
function distance_from_target(ce::AbstractCounterfactualExplanation; K::Int=50, kwrgs...)
    ids = rand(1:size(ce.params[:potential_neighbours], 2), K)
    neighbours = ce.params[:potential_neighbours][:, ids]
    centroid = mean(neighbours; dims=ndims(neighbours))
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
function model_loss_penalty(ce::AbstractCounterfactualExplanation; agg=mean)
    x_ = CounterfactualExplanations.decode_state(ce)
    M = ce.M
    model = isa(M.model, Vector) ? M.model : [M.model]
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
