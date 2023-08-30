"""
    PythonCallExt

Exposes the `PyTorchModel` from the `PythonCallExt` extension.
"""
function PyTorchModel end
export PyTorchModel

"""
    pytorch_model_loader

Exposes the `pytorch_model_loader` function from the `PythonCallExt` extension.
"""
function pytorch_model_loader end
export pytorch_model_loader

"""
    preprocess_python_data

Exposes the `preprocess_python_data` function from the `PythonCallExt` extension.
"""
function preprocess_python_data end
export preprocess_python_data
