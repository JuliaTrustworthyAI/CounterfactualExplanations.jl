using Flux: Flux

"""
    distance(
        cf::AbstractArray,
        from::AbstractArray;
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
    if isnothing(from)
        from = ce.factual
        from = CounterfactualExplanations.encode_state(ce, from)
    end

    if !isnothing(d)
        # Select subset of features:
        selector = Flux.onehotbatch(d, 1:size(cf, 1))
        from = from'selector |> permutedims
        cf = cf'selector |> permutedims
    end

    # Cosine:
    if cosine
        xs = eachslice(cf; dims=ndims(cf))
        δs = map(cf -> cos_dist(cf, from), xs)
        dist = agg(δs)
        return dist
    end

    xs = eachslice(cf; dims=ndims(cf))                                  # slices along the last dimension (i.e. the number of counterfactuals)
    if isnothing(weights)
        dist = agg(map(cf -> LinearAlgebra.norm(cf .- from, p), xs))    # aggregate across counterfactuals
    else
        @assert length(weights) == size(first(xs), ndims(first(xs))) "The length of the weights vector must match the number of features."
        dist = agg(map(cf -> (LinearAlgebra.norm.(cf .- from, p)'weights)[1], xs))   # aggregate across counterfactuals
    end
    return dist
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
    cos_dist(x,y)

Computes the cosine distance between two vectors.
"""
function cos_dist(x, y)
    cos_sim = (x'y / (norm(x) * norm(y)))[1]
    return 1 - cos_sim
end
