module PythonCallExt

using CounterfactualExplanations
using Flux
using PythonCall

### BEGIN utils.jl

"""
    CounterfactualExplanations.pytorch_model_loader(model_path::String, model_file::String, class_name::String, pickle_path::String)

Loads a previously saved PyTorch model.

# Arguments
- `model_path::String`: Path to the directory containing the model file.
- `model_file::String`: Name of the model file.
- `class_name::String`: Name of the model class.
- `pickle_path::String`: Path to the pickle file containing the model.

# Returns
- `model`: The loaded PyTorch model.

# Example
```{julia}
model = pytorch_model_loader(
    "src/models/pretrained/pytorch",
    "pytorch_model.py",
    "PyTorchModel",
    "src/models/pretrained/pytorch/pytorch_model.pt",
)
```
"""
function CounterfactualExplanations.pytorch_model_loader(
    model_path::String, model_file::String, class_name::String, pickle_path::String
)
    sys = PythonCall.pyimport("sys")
    torch = PythonCall.pyimport("torch")

    # Check whether the path is correct
    if !endswith(pickle_path, ".pt")
        throw(ArgumentError("pickle_path must end with '.pt'"))
    end

    # Make sure Python is able to import the module
    if !in(model_path, sys.path)
        sys.path.append(model_path)
    end

    PythonCall.pyimport(model_file => class_name)
    model = torch.load(pickle_path)
    return model
end

"""
    CounterfactualExplanations.preprocess_python_data(data::CounterfactualData)

Converts a `CounterfactualData` object to an input tensor and a label tensor.

# Arguments
- `data::CounterfactualData`: The data to be converted.

# Returns
- `(x_python::Py, y_python::Py)`: A tuple of tensors resulting from the conversion, `x_python` holding the features and `y_python` holding the labels.

# Example
x_python, y_python = preprocess_python_data(counterfactual_data) # converts `counterfactual_data` to tensors `x_python` and `y_python
"""
function CounterfactualExplanations.preprocess_python_data(data::CounterfactualData)
    x_julia = data.X
    y_julia = data.y

    # Convert data to tensors
    torch = PythonCall.pyimport("torch")
    np = PythonCall.pyimport("numpy")

    x_python = Float32.(x_julia)
    x_python = np.array(x_python)
    x_python = torch.tensor(x_python).T

    y_python = Float32.(y_julia)
    y_python = np.array(y_python)
    y_python = torch.tensor(y_python)

    return x_python, y_python
end

### END 

### BEGIN models.jl

using CounterfactualExplanations.Models

"""
PyTorchModel <: AbstractDifferentiableModel

Constructor for models trained in `PyTorch`. 
"""
struct PyTorchModel <: AbstractDifferentiableModel
    model::Any
    likelihood::Symbol
    function PyTorchModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi]`"
                ),
            )
        end
    end
end

"Outer constructor that extends method from parent package."
CounterfactualExplanations.PyTorchModel(args...) = PyTorchModel(args...)

"""
    function Models.logits(M::PyTorchModel, x::AbstractArray)

Calculates the logit scores output by the model `M` for the input data `X`.

# Arguments
- `M::PyTorchModel`: The model selected by the user. Must be a model defined using PyTorch.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The logit scores for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data points in `X`
"""
function Models.logits(M::PyTorchModel, x::AbstractArray)
    torch = PythonCall.pyimport("torch")
    np = PythonCall.pyimport("numpy")

    if !isa(x, Matrix)
        x = reshape(x, length(x), 1)
    end

    ŷ_python = M.model(torch.tensor(np.array(x)).T).detach().numpy()
    ŷ = PythonCall.pyconvert(Matrix, ŷ_python)

    return transpose(ŷ)
end

"""
    function Models.probs(M::PyTorchModel, x::AbstractArray)

Calculates the output probabilities of the model `M` for the input data `X`.

# Arguments
- `M::PyTorchModel`: The model selected by the user. Must be a model defined using PyTorch.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The probabilities for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the probabilities for each output class for the data points in `X`
"""
function Models.probs(M::PyTorchModel, x::AbstractArray)
    if M.likelihood == :classification_binary
        return Flux.σ.(logits(M, x))
    elseif M.likelihood == :classification_multi
        return Flux.softmax(logits(M, x))
    end
end

### END 

### BEGIN generators.jl

using CounterfactualExplanations.Generators

"""
    Generators.∂ℓ(generator::AbstractGradientBasedGenerator, M::PyTorchModel, ce::AbstractCounterfactualExplanation)

Method for computing the gradient of the loss function at the current counterfactual state for gradient-based generators operating on PyTorch models.
The gradients are calculated through PyTorch using PythonCall.jl.

# Arguments
- `generator::AbstractGradientBasedGenerator`: The generator object that is used to generate the counterfactual explanation.
- `M::Models.PyTorchModel`: The PyTorch model for which the counterfactual is generated.
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object for which the gradient is calculated.

# Returns
- `grad::AbstractArray`: The gradient of the loss function at the current counterfactual state.

# Example
grad = ∂ℓ(generator, M, ce) # calculates the gradient of the loss function at the current counterfactual state.
"""
function Generators.∂ℓ(
    generator::AbstractGradientBasedGenerator,
    M::PyTorchModel,
    ce::AbstractCounterfactualExplanation,
)
    torch = PythonCall.pyimport("torch")
    np = PythonCall.pyimport("numpy")

    x = ce.x
    target = Float32.(ce.target_encoded)

    x = torch.tensor(np.array(reshape(x, 1, length(x))))
    x.requires_grad = true

    target = torch.tensor(np.array(reshape(target, 1, length(target))))
    target = target.squeeze()

    output = M.model(x).squeeze()

    obj_loss = torch.nn.BCEWithLogitsLoss()(output, target)
    obj_loss.backward()

    grad = PythonCall.pyconvert(Matrix, x.grad.t().detach().numpy())

    return grad
end

### END 

end
