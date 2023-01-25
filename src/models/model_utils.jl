using Flux
using MLJBase

"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    xs = MLUtils.unstack(X, dims=2)
    output_dim = length(unique(y))
    if output_dim > 2
        y = Flux.onehotbatch(y, sort(unique(y)))
        y = Flux.unstack(y; dims=3)
    end
    data = zip(xs, y)
    return data
end

"""
    model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)

Helper function to compute F-Score for `AbstractFittedModel` on a (test) data set.
"""
function model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(test_data)
    m = MulticlassFScore()
    binary = M.likelihood == :classification_binary
    if binary
        proba = reduce(hcat, map(x -> binary ? [1 - x, x] : x, probs(M, X)))
        ŷ = Flux.onecold(proba, 0:1)
    else
        y = Flux.onecold(y, 1:size(y, 1))
        ŷ = Flux.onecold(probs(M, X), sort(unique(y)))
    end
    fscore = m(ŷ, vec(y))

    return fscore
end

"""
    predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::AbstractArray)

Returns the predicted output label for a given model `M`, data set `counterfactual_data` and input data `X`.
"""
function predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::AbstractArray)
    p = probs(M, X)
    y_levels = counterfactual_data.y_levels
    binary = M.likelihood == :classification_binary
    n_levels = length(y_levels)
    if binary
        idx = Int.(round.(p) .+ 1)
        y = y_levels[idx]
    else
        idx = Flux.onecold(p, 1:n_levels)
        y = y_levels[idx]
    end
    return y
end

"""
    predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData)

Returns the predicted output labels for all data points of data set `counterfactual_data` for a given model `M`.
"""
function predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData)
    X = counterfactual_data.X
    return predict_label(M, counterfactual_data, X)
end