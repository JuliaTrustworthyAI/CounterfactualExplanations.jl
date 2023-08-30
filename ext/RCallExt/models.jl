using CounterfactualExplanations.Models

"""
Base type for differentiable models written in R.
"""
abstract type AbstractRModel <: Models.AbstractDifferentiableModel end

"""
RTorchModel <: AbstractRModel

Constructor for models trained in `R`. 
"""
struct RTorchModel <: AbstractRModel
    nn::Any
    likelihood::Symbol
end

"""
    function logits(M::PyTorchModel, x::AbstractArray)

Calculates the logit scores output by the model `M` for the input data `X`.

# Arguments
- `M::RTorchModel`: The model selected by the user. Must be a model defined using R.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The logit scores for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data points in `X`
"""
function Models.logits(model::RTorchModel, x::AbstractArray)
    if !isa(x, Matrix)
        x = reshape(x, length(x), 1)
    end

    model_nn = model.nn

    ŷ = RCall.rcopy(R"as_array($model_nn(torch_tensor(t($x))))")
    ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]

    return transpose(ŷ)
end

"""
    function probs(M::RTorchModel, x::AbstractArray)

Calculates the output probabilities of the model `M` for the input data `X`.

# Arguments
- `M::RTorchModel`: The model selected by the user. Must be a model defined using R.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The probabilities for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the probabilities for each output class for the data points in `X`
"""
function Models.probs(model::RTorchModel, x::AbstractArray)
    if model.likelihood == :classification_binary
        return Flux.σ.(logits(model, x))
    elseif model.likelihood == :classification_multi
        return Flux.softmax(logits(model, x))
    end
end
