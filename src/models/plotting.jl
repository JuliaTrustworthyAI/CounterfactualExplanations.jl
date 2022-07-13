using Plots
import Plots: plot

function plot(M::AbstractFittedModel,data::DataPreprocessing.CounterfactualData; 
    colorbar=true,title="",length_out=50,zoom=-1,xlim=nothing,ylim=nothing,linewidth=0.1,alpha=1.0,kwargs...)
    
    X, y = DataPreprocessing.unpack(data)
    X = X'
    y = vec(y)

    # Surface range:
    if isnothing(xlim)
        xlim = (minimum(X[:,1]),maximum(X[:,1])).+(zoom,-zoom)
    else
        xlim = xlim .+ (zoom,-zoom)
    end
    if isnothing(ylim)
        ylim = (minimum(X[:,2]),maximum(X[:,2])).+(zoom,-zoom)
    else
        ylim = ylim .+ (zoom,-zoom)
    end
    x_range = collect(range(xlim[1],stop=xlim[2],length=length_out))
    y_range = collect(range(ylim[1],stop=ylim[2],length=length_out))
    Z = [Models.probs(M,[x, y])[1] for x=x_range, y=y_range]

    # Contour:
    contourf(
        x_range, y_range, Z'; 
        colorbar=colorbar, title=title, linewidth=linewidth,
        xlim=xlim,
        ylim=ylim,
        kwargs...
    )
    
    # Samples:
    DataPreprocessing.scatter!(data; alpha=alpha)

end