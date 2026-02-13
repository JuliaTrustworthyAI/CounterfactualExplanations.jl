using Flux: Flux
using LinearAlgebra: LinearAlgebra, norm, dot

"""
    distance(
        cf::AbstractArray,
        ce::AbstractCounterfactualExplanation;
        from::Union{AbstractArray,Nothing}=nothing,
        agg=mean,
        p::Real=1,
        weights::Union{Nothing,AbstractArray}=nothing,
        cosine::Bool=false,
        d::Union{Nothing,Vector{Int}}=nothing,
    )

Computes the distance of the counterfactual to the original factual.
"""
function distance(
    cf::AbstractArray,
    ce::AbstractCounterfactualExplanation;
    from::Union{AbstractArray,Nothing}=nothing,
    agg=mean,
    p::Real=1,
    weights::Union{Nothing,AbstractArray}=nothing,
    cosine::Bool=false,
    d::Union{Nothing,Vector{Int}}=nothing,
)
    from_data = isnothing(from) ? ce.factual : from

    # Handle feature selection - use direct indexing instead of onehotbatch
    if !isnothing(d)
        from_data = from_data[d, :]
        cf = cf[d, :]
    end

    # Cosine distance
    if cosine
        return _compute_cosine_distance(cf, from_data, agg)
    end

    # Regular distance
    if isnothing(weights)
        return _compute_unweighted_distance(cf, from_data, p, agg)
    else
        return _compute_weighted_distance(cf, from_data, p, weights, agg)
    end
end

"""
    distance(ce::AbstractCounterfactualExplanation; kwrgs...)

Overloads method to be applied directly to `ce`
"""
function distance(ce::AbstractCounterfactualExplanation; kwrgs...)
    cf = CounterfactualExplanations.decode_state(ce)
    return distance(cf, ce; kwrgs...)
end

"""
    cos_dist(x, y)

Computes the cosine distance between two vectors.
"""
function cos_dist(x, y)
    cos_sim = dot(x, y) / (norm(x) * norm(y))
    return 1 - cos_sim
end

# ============================================================================
# Internal helper functions
# ============================================================================

function _compute_cosine_distance(cf::AbstractArray, from::AbstractArray, agg)
    n_samples = size(cf, ndims(cf))
    total = zero(eltype(cf))

    @inbounds for i in 1:n_samples
        cf_slice = selectdim(cf, ndims(cf), i)
        total += cos_dist(cf_slice, from)
    end

    # Handle aggregation
    return if agg === mean
        total / n_samples
    else
        agg([selectdim(cf, ndims(cf), i) for i in 1:n_samples])
    end
end

function _compute_unweighted_distance(cf::AbstractArray, from::AbstractArray, p::Real, agg)
    n_samples = size(cf, ndims(cf))

    # Fast path for mean aggregation (most common case)
    if agg === mean
        total = zero(eltype(cf))
        @inbounds for i in 1:n_samples
            cf_slice = selectdim(cf, ndims(cf), i)
            total += norm(cf_slice .- from, p)
        end
        return total / n_samples
    else
        # Generic aggregation - still avoid eachslice
        distances = Vector{eltype(cf)}(undef, n_samples)
        @inbounds for i in 1:n_samples
            cf_slice = selectdim(cf, ndims(cf), i)
            distances[i] = norm(cf_slice .- from, p)
        end
        return agg(distances)
    end
end

function _compute_weighted_distance(
    cf::AbstractArray, from::AbstractArray, p::Real, weights::AbstractArray, agg
)
    n_samples = size(cf, ndims(cf))
    n_features = size(cf, 1)

    @assert length(weights) == n_features "The length of the weights vector must match the number of features."

    # Fast path for mean aggregation
    if agg === mean
        total = zero(eltype(cf))
        @inbounds for i in 1:n_samples
            weighted_sum = zero(eltype(cf))
            for j in 1:n_features
                diff = cf[j, i] - from[j]
                weighted_sum += abs(diff)^p * weights[j]
            end
            total += weighted_sum^(1 / p)
        end
        return total / n_samples
    else
        # Generic aggregation
        distances = Vector{eltype(cf)}(undef, n_samples)
        @inbounds for i in 1:n_samples
            weighted_sum = zero(eltype(cf))
            for j in 1:n_features
                diff = cf[j, i] - from[j]
                weighted_sum += abs(diff)^p * weights[j]
            end
            distances[i] = weighted_sum^(1 / p)
        end
        return agg(distances)
    end
end
