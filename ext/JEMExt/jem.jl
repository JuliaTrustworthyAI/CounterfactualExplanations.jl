using CounterfactualExplanations.Models
using Distributions
using Flux
using MLJBase
using Tables: columntable
using TaijaBase: TaijaBase

"""
    CounterfactualExplanations.JEM(
        model::JointEnergyModels.JointEnergyClassifier; likelihood::Symbol=:classification_binary
    )

Outer constructor for a neural network with Laplace Approximation from `LaplaceRedux.jl`.
"""
function CounterfactualExplanations.JEM(
    model::JointEnergyModels.JointEnergyClassifier;
    likelihood::Symbol=:classification_binary,
)
    return Models.Model(model, CounterfactualExplanations.JEM(); likelihood=likelihood)
end

"""
    (M::Model)(data::CounterfactualData, type::JEM; kwargs...)
    
Constructs a differentiable tree-based model for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData, type::CounterfactualExplanations.JEM; kwargs...
)
    n = CounterfactualExplanations.DataPreprocessing.outdim(data)
    𝒟y = Categorical(ones(n) ./ n)
    𝒟x = Normal()
    input_dim = size(data.X, 1)
    sampler = JointEnergyModels.ConditionalSampler(
        𝒟x, 𝒟y; input_size=(input_dim,), batch_size=50
    )
    model = JointEnergyModels.JointEnergyClassifier(sampler; kwargs...)
    M = CounterfactualExplanations.JEM(model; likelihood=data.likelihood)
    return M
end

"""
    train(M::JEM, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::JEM`: The wrapper for an JEM model.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::JEM`: The fitted JEM model.
"""
function Models.train(
    M::Models.Model,
    type::CounterfactualExplanations.JEM,
    data::CounterfactualData;
    kwargs...,
)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    if M.likelihood ∉ [:classification_multi, :classification_binary]
        y = float.(y.refs)
    end
    X = columntable(X)
    mach = MLJBase.machine(M.model, X, y)
    MLJBase.fit!(mach)
    Flux.testmode!(mach.fitresult[1])
    M.fitresult = Models.Fitresult(mach.fitresult, Dict())
    return M
end

"""
    Models.logits(M::JEM, X::AbstractArray)

Calculates the logit scores output by the model M for the input data X.

# Arguments
- `M::JEM`: The model selected by the user. Must be a model from the MLJ library.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::Matrix`: A matrix of logits for each output class for each data point in X.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data point x
"""
function Models.logits(
    M::Models.Model, type::CounterfactualExplanations.JEM, X::AbstractArray
)
    nn = M.fitresult()[1][1]
    return nn(X)
end

"""
    Models.probs(
        M::Models.Model,
        type::CounterfactualExplanations.JEM,
        X::AbstractArray,
    )

Overloads the [probs](@ref) method for NeuroTree models.
"""
function Models.probs(
    M::Models.Model, type::CounterfactualExplanations.JEM, X::AbstractArray
)
    if ndims(X) == 1
        X = X[:, :]      # account for 1-dimensional inputs
    end
    return softmax(logits(M, X))
end
