"""
    undersample(data::CounterfactualData, n::Int)

Helper function to randomly undersample `data::CounterfactualData`.
"""
function undersample(data::CounterfactualData, n::Int)

    X, y = unpack_data(data)
    n_classes = length(unique(y))
    n_per_class = Int(round(n / n_classes))
    if n_classes > 2
        y_cls = Flux.onecold(y, 1:n_classes)
    else
        y_cls = y
    end
    classes_ = sort(unique(y_cls))

    idx = sort(reduce(vcat, [sample(findall(vec(y_cls .== cls)), n_per_class, replace=false) for cls in classes_]))
    data.X = X[:, idx]
    data.y = y[:, idx]

    return data

end