using CounterfactualExplanations
using Flux
using MLDatasets
using MLJBase

"""
    load_mnist()

Loads and prepares MNIST data.
"""
function load_mnist(n::Union{Nothing,Int}=nothing)
    X, y = MNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(-1.0, 1.0), standardize=false)
    counterfactual_data.X = Float32.(counterfactual_data.X)
    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
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
    X = X .* 2.0f0 .- 1.0f0
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(-1.0, 1.0))
    counterfactual_data.X = Float32.(counterfactual_data.X)
    return counterfactual_data
end

"""
    load_fashion_mnist(n::Union{Nothing,Int}=nothing)

Loads and prepares FashionMNIST data.
"""
function load_fashion_mnist(n::Union{Nothing,Int}=nothing)
    X, y = FashionMNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(-1.0, 1.0), standardize=false)
    counterfactual_data.X = Float32.(counterfactual_data.X)
    # Undersample:
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(
            counterfactual_data, n
        )
    end
    return counterfactual_data
end

"""
    load_fashion_mnist_test()

Loads and prepares FashionMNIST test data.
"""
function load_fashion_mnist_test()
    X, y = FashionMNIST(:test)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(-1.0, 1.0))
    counterfactual_data.X = Float32.(counterfactual_data.X)
    return counterfactual_data
end