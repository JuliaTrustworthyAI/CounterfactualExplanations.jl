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
    x′ = CounterfactualExplanations.decode_state(ce)
    if ce.num_counterfactuals == 1
        return LinearAlgebra.norm(x′ .- from, p)
    else
        xs = eachslice(x′; dims=ndims(x′))                      # slices along the last dimension (i.e. the number of counterfactuals)
        if isnothing(weights)
            Δ = agg(map(x′ -> LinearAlgebra.norm(x′ .- from, p), xs))            # aggregate across counterfactuals
        else
            @assert length(weights) == size(first(xs), ndims(first(xs))) "The length of the weights vector must match the number of features."
            Δ = agg(map(x′ -> (LinearAlgebra.norm.(x′ .- from, p)'weights)[1], xs))   # aggregate across counterfactuals
        end
        return Δ
    end
end

"""
    distance_from_target(
        ce::AbstractCounterfactualExplanation;
        K::Int=50
    )

Computes the distance of the counterfactual from a point in the target main.
"""
function distance_from_target(ce::AbstractCounterfactualExplanation; K::Int=50, kwrgs...)
    if !(:potential_neighbors ∈ collect(keys(ce.search)))
        ce.search[:potential_neighbors] = find_potential_neighbors(ce)
    end

    ids = rand(1:size(ce.search[:potential_neighbors], 2), K)
    neighbours = ce.search[:potential_neighbors][:, ids]
    centroid = Statistics.mean(neighbours; dims=ndims(neighbours))
    Δ = distance(ce; from=centroid, kwrgs...)
    return Δ
end

function find_potential_neighbors(ce::AbstractCounterfactualExplanation)
    ids = findall(Models.predict_label(ce.M, ce.data) .== ce.target)
    n_candidates = minimum([size(ce.data.y, 2), 1000])
    candidates = DataPreprocessing.select_factual(ce.data, rand(ids, n_candidates))
    potential_neighbors = hcat([x[1] for x in candidates]...)
    return potential_neighbors
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
