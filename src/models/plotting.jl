using NearestNeighborModels, MLJ
function voronoi(X::AbstractMatrix, y::AbstractVector)
    knnc = KNNClassifier(K=1) # KNNClassifier instantiation
    X = MLJ.table(X)
    y = categorical(y)
    knnc_mach = machine(knnc, X, y) 
    fit!(knnc_mach)
    return knnc_mach, y
end

using Plots
import Plots: plot
using Flux
function plot(
    M::AbstractFittedModel,data::DataPreprocessing.CounterfactualData;
    target::Union{Nothing,Int}=nothing,
    colorbar=true,title="",length_out=50,zoom=-1,xlim=nothing,ylim=nothing,linewidth=0.1,alpha=1.0,
    kwargs...
)
    
    X, _ = DataPreprocessing.unpack(data)
    ŷ = Models.probs(M, X) # true predictions
    if size(ŷ,1) > 1
        ŷ = Flux.onecold(ŷ, 1:size(ŷ,1))
    end
    
    X, y, multi_dim = DataPreprocessing.prepare_for_plotting(data)
    
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
    x_range = range(xlim[1],stop=xlim[2],length=length_out)
    y_range = range(ylim[1],stop=ylim[2],length=length_out)

    if multi_dim
        knn1, y_train = voronoi(X, ŷ)
        predict_ = (X::AbstractVector) -> vec(pdf(predict(knn1, MLJ.table(reshape(X,1,2))), levels(y_train)))
        Z = [predict_([x,y]) for x=x_range, y=y_range]
    else
        predict_ = function(X::AbstractVector) 
            z = Models.probs(M,X)
            if length(z) == 1 # binary
                z = [1.0 - z[1], z[1]]
            end
            return z
        end
        Z = [predict_([x,y]) for x=x_range, y=y_range]
    end

    Z = reduce(hcat, Z)

    if isnothing(target)
        if size(Z, 1) > 2
            @info "No target label supplied, using first."
            target = 1
        else
            target = 2
        end
    end
    
    # Contour:
    contourf(
        x_range, y_range, Z[Int(target),:]; 
        colorbar=colorbar, title=title, linewidth=linewidth,
        xlim=xlim,
        ylim=ylim,
        kwargs...
    )
    
    # Samples:
    DataPreprocessing.scatter!(data; alpha=alpha)

end