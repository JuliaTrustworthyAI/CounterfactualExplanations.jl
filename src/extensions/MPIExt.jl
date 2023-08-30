"""
    MPIParallelizer(args...)

Exposes the `MPIParallelizer` function from the `MPIExt` extension.
"""
function MPIParallelizer(args...)
    ext_sym = :MPIExt
    ext = Base.get_extension(@__MODULE__(), ext_sym)
    if !isnothing(ext)
        return ext.MPIParallelizer(args...)
    else
        throw(ArgumentError("Extension $ext_sym not loaded."))
    end
end
export MPIParallelizer
