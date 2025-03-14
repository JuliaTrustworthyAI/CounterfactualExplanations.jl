using Flux: Flux

"""
    distance(
        ce::AbstractCounterfactualExplanation;
        from::Union{Nothing,AbstractArray}=nothing,
        agg=mean,
        p::Real=1,
        weights::Union{Nothing,AbstractArray}=nothing,
    )

Computes the distance of the counterfactual to the original factual.
"""
function distance(
    ce::AbstractCounterfactualExplanation;
    from::Union{Nothing,AbstractArray}=nothing,
    agg=mean,
    p::Real=1,
    weights::Union{Nothing,AbstractArray}=nothing,
    cosine::Bool=false,
    d::Union{Nothing,Vector{Int}}=nothing,
)
    if isnothing(from)
        from = CounterfactualExplanations.factual(ce)
    end
    cf = CounterfactualExplanations.decode_state(ce)
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
        Δ = agg(δs)
        return Δ
    end

    if num_counterfactuals(ce) == 1
        return LinearAlgebra.norm(cf .- from, p)
    else
        xs = eachslice(cf; dims=ndims(cf))                      # slices along the last dimension (i.e. the number of counterfactuals)
        if isnothing(weights)
            Δ = agg(map(cf -> LinearAlgebra.norm(cf .- from, p), xs))            # aggregate across counterfactuals
        else
            @assert length(weights) == size(first(xs), ndims(first(xs))) "The length of the weights vector must match the number of features."
            Δ = agg(map(cf -> (LinearAlgebra.norm.(cf .- from, p)'weights)[1], xs))   # aggregate across counterfactuals
        end
        return Δ
    end
end

"""
    cos_dist(x,y)

Computes the cosine distance between two vectors.
"""
function cos_dist(x, y)
    cos_sim = (x'y / (norm(x) * norm(y)))[1]
    return 1 - cos_sim
end
