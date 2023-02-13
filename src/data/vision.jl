using CounterfactualExplanations
using Flux
using MLDatasets
using MLJBase

"""
    load_mnist()

Loads and prepares MNIST data.
"""
function load_mnist(n::Union{Nothing,Int} = nothing)
    X, y = MNIST(:train)[:]
    X = Flux.flatten(X)
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain = (0, 1))
    # Undersample:
    if !isnothing(n)
        counterfactual_data =
            CounterfactualExplanations.DataPreprocessing.undersample(counterfactual_data, n)
    end
    return counterfactual_data
end

"""
    load_mnist_test()

Loads and prepares MNIST test data.
"""
function load_mnist_test()
    X, y = MNIST(:test)[:]
    X = Flux.flatten(X)
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain = (0, 1))
    return counterfactual_data
end
