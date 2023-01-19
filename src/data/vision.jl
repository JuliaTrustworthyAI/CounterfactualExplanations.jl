using CounterfactualExplanations
using MLDatasets
using MLJBase

function load_mnist()
    X, y = MNIST(:train)[:]
    X = Flux.flatten(X)
    y = categorical(y)
    counterfactual_data = CounterfactualData(X,y)
    return counterfactual_data
end