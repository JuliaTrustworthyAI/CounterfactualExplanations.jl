"""
    pytorch_model_loader(model_path::String, model_file::String, class_name::String, pickle_path::String)

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
function pytorch_model_loader(
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
    preprocess_python_data(data::CounterfactualData)

Converts a `CounterfactualData` object to an input tensor and a label tensor.

# Arguments
- `data::CounterfactualData`: The data to be converted.

# Returns
- `(x_python::Py, y_python::Py)`: A tuple of tensors resulting from the conversion, `x_python` holding the features and `y_python` holding the labels.

# Example
x_python, y_python = preprocess_python_data(counterfactual_data) # converts `counterfactual_data` to tensors `x_python` and `y_python
"""
function preprocess_python_data(data::CounterfactualData)
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
