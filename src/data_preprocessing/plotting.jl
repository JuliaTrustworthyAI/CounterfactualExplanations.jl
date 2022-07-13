using Plots
import Plots: scatter!
using TSne
function scatter!(data::CounterfactualData; kwargs...)
    X, y = unpack(data)
    @assert size(X,2) != 1 "Don't know how to plot 1-dimensional data."
    if size(X,2) > 2
        @info "Using t-SNE to embed data into two dimenions for plotting."
        X = tsne(X',2)'
    end
    Plots.scatter!(X[1,:],X[2,:],group=Int.(vec(y)),color=Int.(vec(y)); kwargs...)
end