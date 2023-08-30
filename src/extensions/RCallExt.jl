"""
    RTorchModel(args...)

Exposes the `RTorchModel` function from the `RCallExt` extension.
"""
function RTorchModel(args...)
    ext_sym = :RCallExt
    ext = Base.get_extension(@__MODULE__(), ext_sym)
    if !isnothing(ext)
        return ext.RTorchModel(args...)
    else
        throw(ArgumentError("Extension $ext_sym not loaded."))
    end
end

"""
    rtorch_model_loader(args...)

Exposes the `rtorch_model_loader` function from the `RCallExt` extension.
"""
function rtorch_model_loader(args...)
    ext_sym = :RCallExt
    ext = Base.get_extension(@__MODULE__(), ext_sym)
    if !isnothing(ext)
        return ext.rtorch_model_loader(args...)
    else
        throw(ArgumentError("Extension $ext_sym not loaded."))
    end
end
export rtorch_model_loader
