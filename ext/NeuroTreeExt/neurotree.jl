using CounterfactualExplanations.Models
using Flux
using MLJBase
using Tables: columntable

"Type union for NeuroTree classifiers and regressors."
const AtomicNeuroTree = Union{
    NeuroTreeModels.NeuroTreeClassifier,NeuroTreeModels.NeuroTreeRegressor
}

"""
    CounterfactualExplanations.NeuroTree(
        model::AtomicNeuroTree; likelihood::Symbol=:classification_binary
    )

Outer constructor for a differentiable tree-based model from `NeuroTreeModels.jl`.
"""
function CounterfactualExplanations.NeuroTree(
    model::AtomicNeuroTree; likelihood::Symbol=:classification_binary
)
    return Models.Model(
        model, CounterfactualExplanations.NeuroTree(); likelihood=likelihood
    )
end

"""
    (M::Model)(data::CounterfactualData, type::NeuroTree; kwargs...)
    
Constructs a differentiable tree-based model for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData, type::CounterfactualExplanations.NeuroTree; kwargs...
)
    model = NeuroTreeModels.NeuroTreeClassifier(; kwargs...)
    return NeuroTree(model; likelihood=data.likelihood)
end

"""
    train(M::NeuroTreeModel, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::NeuroTreeModel`: The wrapper for an NeuroTree model.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::NeuroTreeModel`: The fitted NeuroTree model.
"""
function Models.train(
    M::Models.Model, type::CounterfactualExplanations.NeuroTree, data::CounterfactualData
)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    if M.likelihood ∉ [:classification_multi, :classification_binary]
        y = float.(y.refs)
    end
    X = columntable(X)
    mach = MLJBase.machine(M.model, X, y)
    MLJBase.fit!(mach)
    Flux.testmode!(mach.fitresult.chain)
    M.fitresult = mach.fitresult
    return M
end

"""
    Models.logits(M::NeuroTreeModel, X::AbstractArray)

Calculates the logit scores output by the model M for the input data X.

# Arguments
- `M::NeuroTreeModel`: The model selected by the user. Must be a model from the MLJ library.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::Matrix`: A matrix of logits for each output class for each data point in X.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data point x
"""
function Models.logits(
    M::Models.Model, type::CounterfactualExplanations.NeuroTree, X::AbstractArray
)
    return M.fitresult(X)
end

"""
    Models.probs(
        M::Models.Model,
        type::CounterfactualExplanations.NeuroTree,
        X::AbstractArray,
    )

Overloads the [probs](@ref) method for NeuroTree models.
"""
function Models.probs(
    M::Models.Model, type::CounterfactualExplanations.NeuroTree, X::AbstractArray
)
    if ndims(X) == 1
        X = X[:, :]      # account for 1-dimensional inputs
    end
    return softmax(logits(M, X))
end
