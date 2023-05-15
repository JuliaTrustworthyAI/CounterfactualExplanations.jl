using CounterfactualExplanations
using Flux
using MLDatasets
using MLJBase

import MLUtils: flatten

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

"""
    load_cifar_10(n::Union{Nothing, Int}=nothing)

Loads and preprocesses data from the CIFAR-10 dataset for use in counterfactual explanations.

# Arguments
- `n::Union{Nothing, Int}=nothing`: The number of samples to subsample from the dataset. If `n` is not specified, all samples will be used.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed data.

# Example
data = load_cifar_10(1000) # loads and preprocesses 1000 samples from the CIFAR-10 dataset

"""
function load_cifar_10(n::Union{Nothing,Int}=nothing)
    X, y = CIFAR10()[:] # [:] gives us X, y
    X = flatten(X)
    X = X .* 2 .- 1 # normalization between [-1, 1]
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(-1.0, 1.0), standardize=false)
    if !isnothing(n)
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.subsample(counterfactual_data, n)
    end
    return counterfactual_data
end

"""
    load_cifar_10_test()

Loads and preprocesses test data from the CIFAR-10 dataset for use in counterfactual explanations.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed test data.

# Example
test_data = load_cifar_10_test() # loads and preprocesses test data from the CIFAR-10 dataset

"""
function load_cifar_10_test()
    X, y = CIFAR10(:test)[:]
    X = flatten(X)
    X = X .* 2 .- 1 # normalization between [-1, 1]
    y = categorical(y)
    counterfactual_data = CounterfactualData(X, y; domain=(-1.0, 1.0))
    return counterfactual_data
end