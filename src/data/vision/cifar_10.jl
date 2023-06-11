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
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; domain=(-1.0, 1.0), standardize=false
    )
    if !isnothing(n)
        counterfactual_data = subsample(counterfactual_data, n)
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
    counterfactual_data = CounterfactualExplanations.CounterfactualData(
        X, y; domain=(-1.0, 1.0)
    )
    return counterfactual_data
end
