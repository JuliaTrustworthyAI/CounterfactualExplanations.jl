struct NotImplementedModel <: Exception
    M::AbstractModel
end

function Base.showerror(io::IO, e::NotImplementedModel)
    return print(io, "Method not implemented for model type ", e.M)
end
