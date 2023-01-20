using Flux

"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    xs = MLUtils.unstack(X,dims=2) 
    output_dim = length(unique(y))
    if output_dim > 2
        y = Flux.onehotbatch(y, sort(unique(y)))
        y = Flux.unstack(y,3)
    end
    data = zip(xs,y)
    return data
end