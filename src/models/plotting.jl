using Flux
using MLJBase
using NearestNeighborModels
using Plots

function voronoi(X::AbstractMatrix, y::AbstractVector)
    knnc = KNNClassifier(K = 1) # KNNClassifier instantiation
    X = MLJBase.table(X)
    y = categorical(y)
    knnc_mach = machine(knnc, X, y)
    MLJBase.fit!(knnc_mach)
    return knnc_mach, y
end

function Plots.plot(
    M::AbstractFittedModel,
    data::DataPreprocessing.CounterfactualData;
    target::Union{Nothing,TargetType} = nothing,
    colorbar = true,
    title = "",
    length_out = 50,
    zoom = -1,
    xlims = nothing,
    ylims = nothing,
    linewidth = 0.1,
    alpha = 1.0,
    dim_red::Symbol = :pca,
    kwargs...,
)

    X, _ = DataPreprocessing.unpack(data)
    ŷ = Models.probs(M, X) # true predictions
    if size(ŷ, 1) > 1
        ŷ = vec(Flux.onecold(ŷ, 1:size(ŷ, 1)))
    else
        ŷ = vec(ŷ)
    end

    X, y, multi_dim = DataPreprocessing.prepare_for_plotting(data; dim_red = dim_red)

    # Surface range:
    if isnothing(xlims)
        xlims = (minimum(X[:, 1]), maximum(X[:, 1])) .+ (zoom, -zoom)
    else
        xlims = xlims .+ (zoom, -zoom)
    end
    if isnothing(ylims)
        ylims = (minimum(X[:, 2]), maximum(X[:, 2])) .+ (zoom, -zoom)
    else
        ylims = ylims .+ (zoom, -zoom)
    end
    x_range = range(xlims[1], stop = xlims[2], length = length_out)
    y_range = range(ylims[1], stop = ylims[2], length = length_out)

    if multi_dim
        knn1, y_train = voronoi(X, ŷ)
        predict_ =
            (X::AbstractVector) -> vec(
                pdf(
                    MLJBase.predict(knn1, MLJBase.table(reshape(X, 1, 2))),
                    levels(y_train),
                ),
            )
        Z = [predict_([x, y]) for x in x_range, y in y_range]
    else
        predict_ = function (X::AbstractVector)
            z = Models.probs(M, X)
            if length(z) == 1 # binary
                z = [1.0 - z[1], z[1]]
            end
            return z
        end
        Z = [predict_([x, y]) for x in x_range, y in y_range]
    end

    Z = reduce(hcat, Z)

    if isnothing(target)
        if size(Z,1) > 2
            @info "No target label supplied, using first."
            target = 1
        else
            target = 2      # assuming binary case [0,1], choose 1 as target
        end
    end

    # Contour:
    contourf(
        x_range,
        y_range,
        Z[Int(target), :];
        colorbar = colorbar,
        title = title,
        linewidth = linewidth,
        xlims = xlims,
        ylims = ylims,
        kwargs...,
    )

    # Samples:
    scatter!(data; dim_red = dim_red, alpha = alpha, kwargs...)

end
