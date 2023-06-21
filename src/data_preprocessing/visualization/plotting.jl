_n_neighbors(tfn::UMAP.UMAP_) = size(tfn.knns, 1)

function MultivariateStats.predict(tfn::UMAP.UMAP_, X::AbstractArray)
    n_neighbors = minimum([_n_neighbors(tfn), size(X, 2) - 1])
    if n_neighbors == 0
        # A simple catch in case n_samples = 1: add random sample and remove afterwards.
        X = hcat(X, rand(size(X, 1)))
        n_neighbors = 1
        X = UMAP.transform(tfn, X; n_neighbors=n_neighbors)
        X = X[:, 1]
    else
        X = UMAP.transform(tfn, X; n_neighbors=n_neighbors)
    end
    return X
end

function embed(data::CounterfactualData, X::AbstractArray=nothing; dim_red::Symbol=:pca)

    # Training compressor:
    if isnothing(data.compressor)
        X_train, _ = unpack_data(data)
        if size(X_train, 1) < 3
            tfn = data.compressor
        else
            @info "Training model to compress data."
            if dim_red == :umap
                n_neighbors = minimum([size(X_train, 2) - 1, 5])
                tfn = UMAP_(X_train, 2; n_neighbors=n_neighbors)
            end
            if dim_red == :pca
                tfn = MultivariateStats.fit(PCA, X_train; maxoutdim=2)
            end
            data.compressor = tfn
            X = isnothing(X) ? X_train : X
        end
    else
        tfn = data.compressor
    end

    # Transforming:
    if !isnothing(tfn) && !isnothing(X)
        X = MultivariateStats.predict(tfn, X)
    else
        X = isnothing(X) ? X_train : X
    end
    return X
end

function prepare_for_plotting(data::CounterfactualData; dim_red::Symbol=:pca)
    X, _ = unpack_data(data)
    y = data.output_encoder.labels
    @assert size(X, 1) != 1 "Don't know how to plot 1-dimensional data."
    multi_dim = size(X, 1) > 2
    if multi_dim
        X = embed(data, X; dim_red=dim_red)
    end
    return X', y, multi_dim
end

function Plots.scatter!(data::CounterfactualData; dim_red::Symbol=:pca, kwargs...)
    X, y, _ = prepare_for_plotting(data; dim_red=dim_red)
    _c = Int.(y.refs)
    return Plots.scatter!(X[:, 1], X[:, 2]; group=y, colour=_c, kwargs...)
end
