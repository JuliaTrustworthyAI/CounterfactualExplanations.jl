using ..CounterfactualExplanations
using LinearAlgebra
using SliceMap
using Statistics: mean

"""
    distance(counterfactual_explanation::AbstractCounterfactualExplanation, p::Real=2)

Computes the distance of the counterfactual to the original factual.
"""
function distance(
    counterfactual_explanation::AbstractCounterfactualExplanation, p::Real=2; agg=mean
)
    x = CounterfactualExplanations.factual(counterfactual_explanation)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    Δ = agg(SliceMap.slicemap(_x -> permutedims([norm(_x .- x, p)]), x′; dims=(1, 2)))
    return Δ
end

"""
    distance_l0(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L0 distance of the counterfactual to the original factual.
"""
function distance_l0(
    counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean
)
    return distance(counterfactual_explanation, 0; agg=agg)
end

"""
    distance_l1(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L1 distance of the counterfactual to the original factual.
"""
function distance_l1(
    counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean
)
    return distance(counterfactual_explanation, 1; agg=agg)
end

"""
    distance_l2(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L2 (Euclidean) distance of the counterfactual to the original factual.
"""
function distance_l2(
    counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean
)
    return distance(counterfactual_explanation, 2; agg=agg)
end

"""
    distance_linf(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L-inf distance of the counterfactual to the original factual.
"""
function distance_linf(
    counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean
)
    return distance(counterfactual_explanation, Inf; agg=agg)
end

"""
    ddp_diversity(
        counterfactual_explanation::AbstractCounterfactualExplanation;
        perturbation_size=1e-5
    )

Evaluates how diverse the counterfactuals are using a Determinantal Point Process (DDP).
"""
function ddp_diversity(
    counterfactual_explanation::AbstractCounterfactualExplanation;
    perturbation_size=1e-3,
    agg=det,
)
    X = counterfactual_explanation.s′
    xs = eachslice(X; dims=ndims(X))
    K = [1 / (1 + norm(x .- y)) for x in xs, y in xs]
    K += LinearAlgebra.Diagonal(
        randn(eltype(X), size(X, 3)) * convert(eltype(X), perturbation_size)
    )
    cost = -agg(K)
    return cost
end

"""
    distance_from_target(
        counterfactual_explanation::AbstractCounterfactualExplanation, p::Int=2; 
        agg=mean, K::Int=50
    )

Computes the distance of the counterfactual from a point in the target main.
"""
function distance_from_target(
    counterfactual_explanation::AbstractCounterfactualExplanation,
    p::Int=2;
    agg=mean,
    K::Int=50,
)
    ids = rand(1:size(counterfactual_explanation.params[:potential_neighbours], 2), K)
    neighbours = counterfactual_explanation.params[:potential_neighbours][:, ids]
    centroid = mean(neighbours; dims=2)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    Δ = agg(
        SliceMap.slicemap(_x -> permutedims([norm(_x .- centroid, p)]), x′; dims=(1, 2))
    )
    return Δ
end

"""
    function model_loss_penalty(
        counterfactual_explanation::AbstractCounterfactualExplanation;
        agg=mean
    )

Additional penalty for ClaPROARGenerator.
"""
function model_loss_penalty(
    counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean
)
    x_ = CounterfactualExplanations.decode_state(counterfactual_explanation)
    M = counterfactual_explanation.M
    model = isa(M.model, Vector) ? M.model : [M.model]
    y_ = counterfactual_explanation.target_encoded

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
