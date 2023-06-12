function voronoi(X::AbstractMatrix, y::AbstractVector)
    knnc = NearestNeighborModels.KNNClassifier(; K=1) # KNNClassifier instantiation
    X = MLJBase.table(X)
    y = MLJBase.categorical(y)
    knnc_mach = MLJBase.machine(knnc, X, y)
    MLJBase.fit!(knnc_mach)
    return knnc_mach, y
end