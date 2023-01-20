using Flux
using MLJBase

"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    xs = MLUtils.unstack(X, dims=2)
    output_dim = length(unique(y))
    if output_dim > 2
        y = Flux.onehotbatch(y, sort(unique(y)))
        y = Flux.unstack(y; dims = 3)
    end
    data = zip(xs, y)
    return data
end

"""
    model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)

Helper function to compute F-Score for `AbstractFittedModel` on a (test) data set.
"""
function model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(test_data)
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