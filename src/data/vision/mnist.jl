"""
    load_mnist()

Loads and prepares MNIST data.
"""
function load_mnist(n::Union{Nothing,Int}=nothing)
    X, y = MLDatasets.MNIST(:train)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; domain=(-1.0, 1.0), standardize=false
    )
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
    X, y = MLDatasets.MNIST(:test)[:]
    X = Flux.flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = MLJBase.categorical(y)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; domain=(-1.0, 1.0)
    )
    return counterfactual_data
end
