using Flux
using MLJBase

"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData; batchsize=1)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
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
    predict_proba(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::Union{Nothing,AbstractArray})

Returns the predicted output probabilities for a given model `M`, data set `counterfactual_data` and input data `X`.
"""
function predict_proba(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::Union{Nothing,AbstractArray})
    X = isnothing(X) ? counterfactual_data.X : X
    p = probs(M, X)
    # println(p)
    binary = M.likelihood == :classification_binary
    function binary_to_onehot(p)
        nobs = size(p, 2)
        A = hcat(ones(nobs), zeros(nobs))
        B = [-1 1; 0 0]
        return permutedims(permutedims(p) .* A * B .+ A)
    end
    p = binary ? binary_to_onehot(p) : p
    return p
end

"""
    predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::AbstractArray)

Returns the predicted output label for a given model `M`, data set `counterfactual_data` and input data `X`.
"""
function predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::AbstractArray)
    y_levels = counterfactual_data.y_levels
    p = predict_proba(M, counterfactual_data, X)
    y = Flux.onecold(p, y_levels)
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