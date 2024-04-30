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
    X = ce.data[].X
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