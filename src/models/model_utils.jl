using Flux
using MLJBase
using PythonCall

"""
    data_loader(data::CounterfactualData)

Prepares counterfactual data for training in Flux.
"""
function data_loader(data::CounterfactualData; batchsize=1)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    return DataLoader((X, y); batchsize=batchsize)
end

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
        throw(
                ArgumentError(
                    "pickle_path must end with '.pt'"
                ),
            )
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
    model_evaluation(M::AbstractFittedModel, test_data::CounterfactualData)

Helper function to compute F-Score for `AbstractFittedModel` on a (test) data set.
"""
function model_evaluation(
    M::AbstractFittedModel, test_data::CounterfactualData; measure=multiclass_f1score
)
    measure = measure isa AbstractVector ? measure : [measure]
    y = test_data.output_encoder.y
    ŷ = predict_label(M, test_data)
    _eval = [m(ŷ, vec(y)) for m in measure]
    return _eval
end

"""
    binary_to_onehot(p)

Helper function to turn dummy-encoded variable into onehot-encoded variable.
"""
function binary_to_onehot(p)
    nobs = size(p, 2)
    A = hcat(ones(nobs), zeros(nobs))
    B = [-1 1; 0 0]
    return permutedims(permutedims(p) .* A * B .+ A)
end

"""
    predict_proba(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::Union{Nothing,AbstractArray})

Returns the predicted output probabilities for a given model `M`, data set `counterfactual_data` and input data `X`.
"""
function predict_proba(
    M::AbstractFittedModel,
    counterfactual_data::Union{Nothing,CounterfactualData},
    X::Union{Nothing,AbstractArray},
)
    @assert !(isnothing(counterfactual_data) && isnothing(X))
    X = isnothing(X) ? counterfactual_data.X : X
    p = probs(M, X)
    binary = M.likelihood == :classification_binary
    p = binary ? binary_to_onehot(p) : p
    return p
end

"""
    predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::AbstractArray)

Returns the predicted output label for a given model `M`, data set `counterfactual_data` and input data `X`.
"""
function predict_label(
    M::AbstractFittedModel, counterfactual_data::CounterfactualData, X::AbstractArray
)
    p = predict_proba(M, counterfactual_data, X)
    y = Flux.onecold(p, counterfactual_data.y_levels)
    return y
end

"""
    predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData)

Returns the predicted output labels for all data points of data set `counterfactual_data` for a given model `M`.
"""
function predict_label(M::AbstractFittedModel, counterfactual_data::CounterfactualData)
    X = counterfactual_data.X
    return predict_label(M, counterfactual_data, X)
end
