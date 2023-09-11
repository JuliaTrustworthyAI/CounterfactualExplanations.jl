vectorize_collection(collection::Vector) = collection

vectorize_collection(collection::Base.Iterators.Zip) = map(x -> x[1], collect(collection))

function vectorize_collection(collection::Matrix)
    return [collection]
end

"""
    parallelize(
        parallelizer::nothing,
        f::Function,
        args...;
        kwargs...,
    )

If no `AbstractParallelizer` has been supplied, just call or broadcast the function. 
"""
function parallelize(parallelizer::Nothing, f::Function, args...; verbose::Bool=false, kwargs...)
    collection = args[1]
    collection = vectorize_collection(collection)
    return f.(collection, args[2:end]...; kwargs...)
end
