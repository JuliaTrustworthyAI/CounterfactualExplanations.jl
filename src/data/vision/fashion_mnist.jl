"""
    load_fashion_mnist(n::Union{Nothing,Int}=nothing)

Loads and prepares FashionMNIST data.
"""
function load_fashion_mnist(n::Union{Nothing,Int}=nothing)
    X, y = FashionMNIST(:train)[:]
    X = flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = categorical(y)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; domain=(-1.0, 1.0), standardize=false
    )
    counterfactual_data.X = Float32.(counterfactual_data.X)
    # Undersample:
    if !isnothing(n)
        counterfactual_data = subsample(counterfactual_data, n)
    end
    return counterfactual_data
end

"""
    load_fashion_mnist_test()

Loads and prepares FashionMNIST test data.
"""
function load_fashion_mnist_test()
    X, y = FashionMNIST(:test)[:]
    X = flatten(X)
    X = X .* 2.0f0 .- 1.0f0
    y = categorical(y)
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; domain=(-1.0, 1.0)
    )
    counterfactual_data.X = Float32.(counterfactual_data.X)
    return counterfactual_data
end
