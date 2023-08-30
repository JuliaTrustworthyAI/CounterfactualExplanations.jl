"""
    PythonCallExt

Exposes the `PyTorchModel` from the `PythonCallExt` extension.
"""
function PyTorchModel(args...)
    ext_sym = :PythonCallExt
    ext = Base.get_extension(@__MODULE__(), ext_sym)
    if !isnothing(ext)
        return ext.PyTorchModel(args...)
    else
        throw(ArgumentError("Extension $ext_sym not loaded."))
    end
end

"""
    pytorch_model_loader(args...)

Exposes the `pytorch_model_loader` function from the `PythonCallExt` extension.
"""
function pytorch_model_loader(args...)
    ext_sym = :PythonCallExt
    ext = Base.get_extension(@__MODULE__(), ext_sym)
    if !isnothing(ext)
        return ext.pytorch_model_loader(args...)
    else
        throw(ArgumentError("Extension $ext_sym not loaded."))
    end
end

"""
    preprocess_python_data(args...)

Exposes the `preprocess_python_data` function from the `PythonCallExt` extension.
"""
function preprocess_python_data(args...)
    ext_sym = :PythonCallExt
    ext = Base.get_extension(@__MODULE__(), ext_sym)
    if !isnothing(ext)
        return ext.preprocess_python_data(args...)
    else
        throw(ArgumentError("Extension $ext_sym not loaded."))
    end
end
    
