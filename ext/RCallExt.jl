module RCallExt

using CounterfactualExplanations
using Flux
using RCall

### BEGIN utils.jl

"""
    rtorch_model_loader(model_path::String)

Loads a previously saved R torch model.

# Arguments
- `model_path::String`: Path to the directory containing the model file.

# Returns
- `loaded_model`: Path to the pickle file containing the model.

# Example
```{julia}
model = rtorch_model_loader("dev/R_call_implementation/model.pt")
```
"""
function CounterfactualExplanations.rtorch_model_loader(model_path::String)
    R"""
    library(torch)
    loaded_model <- torch_load($model_path)
    """

    return R"loaded_model"
end

### END

### BEGIN models.jl

using CounterfactualExplanations.Models

"""
RTorchModel <: Models.AbstractDifferentiableModel

Constructor for models trained in `R`. 
"""
struct RTorchModel <: Models.AbstractDifferentiableModel
    nn::Any
    likelihood::Symbol
end

"Outer constructor that extends method from parent package."
CounterfactualExplanations.RTorchModel(args...) = RTorchModel(args...)

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

### END

### BEGIN generators.jl

using CounterfactualExplanations.Generators

"""
    Generators.∂ℓ(
        generator::AbstractGradientBasedGenerator,
        model::RTorchModel,
        ce::AbstractCounterfactualExplanation,
    )

Extends the `∂ℓ` function for gradient-based generators operating on RTorch models.
"""
function Generators.∂ℓ(
    generator::AbstractGradientBasedGenerator,
    model::RTorchModel,
    ce::AbstractCounterfactualExplanation,
)
    x = ce.x
    target = Float32.(ce.target_encoded)

    model_nn = model.nn

    R"""
    x <- torch_tensor(t($x), requires_grad=TRUE)
    target <- torch_tensor(t($target))
    output <- $model_nn(x)
    obj_loss <- nnf_binary_cross_entropy_with_logits(output, target)
    obj_loss$backward()
    """

    grad = RCall.rcopy(R"t(as_array(x$grad))")
    grad = Float32.(grad)

    return grad
end

### END

end
