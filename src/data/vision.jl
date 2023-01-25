using CounterfactualExplanations
using Flux
using MLDatasets
using MLJBase

"""
    load_mnist()

Loads and prepares MNIST data.
"""
function load_mnist()
    X, y = MNIST(:train)[:]
    X = Flux.flatten(X)
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(0,1))
    return counterfactual_data
end