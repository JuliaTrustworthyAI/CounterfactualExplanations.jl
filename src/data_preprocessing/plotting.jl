
using UMAP
_n_neighbors(tfn::UMAP.UMAP_) = size(tfn.knns,1)

function embed(data::CounterfactualData, X::AbstractArray=nothing; seed::Int=123)

    # Setting seed to always embed in same latent space.
    Random.seed!(seed)

    # Training UMAP:
    if isnothing(data.compressor)
        X_train, _ = unpack(data)
        if size(X_train,1) < 3 
            tfn = data.compressor
        else
            @info "Training UMAP to compress data."
            n_neighbors = minimum([size(X_train,2)-1,15])
            tfn = UMAP_(X_train,2;n_neighbors=n_neighbors)
            data.compressor = tfn
        end
    else
        tfn = data.compressor
    end

    # Transforming:
    if !isnothing(tfn)
        n_neighbors=minimum([_n_neighbors(tfn),size(X,2)-1])
        if n_neighbors==0
            # A simple catch in case n_samples = 1: add random sample and remove afterwards.
            X = hcat(X,rand(size(X,1)))
            n_neighbors = 1
            X = transform(tfn,X;n_neighbors=n_neighbors)
            X = X[:,1]
        else
            X = transform(tfn,X;n_neighbors=n_neighbors)
        end
    else 
        if isnothing(X)
            X = X_train
        end
    end
    return X
end

using Random
function prepare_for_plotting(data::CounterfactualData)
    X, y = unpack(data)
    y = vec(y)
    @assert size(X,1) != 1 "Don't know how to plot 1-dimensional data."
    multi_dim = size(X,1) > 2
    if multi_dim
        X = embed(data, X)
    end
    return X', y, multi_dim
end

using Plots
import Plots: scatter!
function scatter!(data::CounterfactualData; kwargs...)
    X, y, _ = prepare_for_plotting(data)
    Plots.scatter!(X[:,1],X[:,2],group=Int.(y),color=Int.(y); kwargs...)
end