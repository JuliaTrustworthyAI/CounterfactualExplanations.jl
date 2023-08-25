vectorize_collection(collection::Vector) = collection

vectorize_collection(collection::Base.Iterators.Zip) = map(x -> x[1], collect(collection))

function vectorize_collection(collection::Matrix)
    @warn "It looks like there is only one observation in the collection. Are you sure you want to parallelize?"
    return [collection]
end