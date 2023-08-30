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
function rtorch_model_loader(model_path::String)
    R"""
    library(torch)
    loaded_model <- torch_load($model_path)
    """

    return R"loaded_model"
end
