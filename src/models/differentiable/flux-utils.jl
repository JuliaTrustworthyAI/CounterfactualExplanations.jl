"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    xs = MLUtils.unstack(X,dims=2) 
    data = zip(xs,y)
    return data
end