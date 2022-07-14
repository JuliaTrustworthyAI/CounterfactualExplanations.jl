
using TSne
using Random
function prepare_for_plotting(data::CounterfactualData)
    Random.seed!(123)
    X, y = unpack(data)
    X, y = (X', vec(y))
    @assert size(X,2) != 1 "Don't know how to plot 1-dimensional data."
    multi_dim = size(X,2) > 2
    if multi_dim
        @info "Using t-SNE to embed data into two dimenions for plotting."
        X = tsne(X,2)
    end
    return X, y, multi_dim
end

using Plots
import Plots: scatter!
function scatter!(data::CounterfactualData; kwargs...)
    X, y, _ = prepare_for_plotting(data)
    Plots.scatter!(X[:,1],X[:,2],group=Int.(y),color=Int.(y); kwargs...)
end