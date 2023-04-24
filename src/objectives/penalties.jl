using ..CounterfactualExplanations
using LinearAlgebra
using SliceMap
using Statistics: mean

"""
    distance(ce::AbstractCounterfactualExplanation, p::Real=2)

Computes the distance of the counterfactual to the original factual.
"""
function distance(ce::AbstractCounterfactualExplanation, p::Real=2; agg=mean)
    x = CounterfactualExplanations.factual(ce)
    x′ = CounterfactualExplanations.counterfactual(ce)
    Δ = agg(norm(x′ .- x))
    return Δ
end

"""
    distance_l0(ce::AbstractCounterfactualExplanation)

Computes the L0 distance of the counterfactual to the original factual.
"""
function distance_l0(ce::AbstractCounterfactualExplanation; agg=mean)
    return distance(ce, 0; agg=agg)
end

"""
    distance_l1(ce::AbstractCounterfactualExplanation)

Computes the L1 distance of the counterfactual to the original factual.
"""
function distance_l1(ce::AbstractCounterfactualExplanation; agg=mean)
    return distance(ce, 1; agg=agg)
end

"""
    distance_l2(ce::AbstractCounterfactualExplanation)

Computes the L2 (Euclidean) distance of the counterfactual to the original factual.
"""
function distance_l2(ce::AbstractCounterfactualExplanation; agg=mean)
    return distance(ce, 2; agg=agg)
end

"""
    distance_linf(ce::AbstractCounterfactualExplanation)

Computes the L-inf distance of the counterfactual to the original factual.
"""
function distance_linf(ce::AbstractCounterfactualExplanation; agg=mean)
    return distance(ce, Inf; agg=agg)
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
function distance_from_target(
    ce::AbstractCounterfactualExplanation, p::Int=2; agg=mean, K::Int=50
)
    ids = rand(1:size(ce.params[:potential_neighbours], 2), K)
    neighbours = ce.params[:potential_neighbours][:, ids]
    centroid = mean(neighbours; dims=ndims(neighbours))
    x′ = CounterfactualExplanations.counterfactual(ce)
    Δ = agg(norm(x′ .- centroid, p))
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
