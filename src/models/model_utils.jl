using Flux
using MLJBase

"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData; batchsize=1)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    output_dim = length(unique(y))
    if output_dim > 2
        y = data.output_encoder.y   # get raw outputs
        y = Flux.onehotbatch(y, data.y_levels)
    end
    return DataLoader((X, y), batchsize=batchsize)
end

"""
    model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)

Helper function to compute F-Score for `AbstractFittedModel` on a (test) data set.
"""
function model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)
    y = test_data.output_encoder.y
    m = MulticlassFScore()
    ŷ = predict_label(M, test_data)
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