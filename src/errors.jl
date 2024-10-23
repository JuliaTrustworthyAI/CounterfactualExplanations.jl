struct NotImplementedModel <: Exception 
    M::AbstractModel
end

Base.showerror(io::IO, e::NotImplementedModel) = print(io, "Method not implemented for model type ", e.M)