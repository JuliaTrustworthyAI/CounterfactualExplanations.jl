"""
    _subset(data::CounterfactualData, idx::Vector{Int})

Creates a subset of the `data`.
"""
function _subset(data::CounterfactualData, idx::Vector{Int})
    dsub = deepcopy(data)
    dsub.X = dsub.X[:, idx]
    dsub.y = dsub.y[:, idx]
    dsub.output_encoder.y = data.output_encoder.y[idx]
    return dsub
end

"""
    train_test_split(data::CounterfactualData;test_size=0.2)

Splits data into train and test split.
"""
function train_test_split(data::CounterfactualData; test_size=0.2)
    N = size(data.X, 2)
    classes_ = data.y_levels
    n_per_class = round(N / length(classes_))
    y = data.output_encoder.y
    test_idx = sort(reduce(vcat, [sample(findall(vec(y .== cls)), Int(floor(test_size * n_per_class)), replace=false) for cls in classes_]))
    train_idx = setdiff(1:N, test_idx)
    train_data = _subset(data, train_idx)
    test_data = _subset(data, test_idx)
    return train_data, test_data
end

"""
    undersample(data::CounterfactualData, n::Int)

Helper function to randomly undersample `data::CounterfactualData`.
"""
function undersample(data::CounterfactualData, n::Int)

    X, y = unpack_data(data)
    classes_ = data.y_levels
    n_classes = length(classes_)
    n_per_class = Int(round(n / n_classes))
    y_cls = data.output_encoder.labels

    idx = sort(reduce(vcat, [sample(findall(vec(y_cls .== cls)), n_per_class, replace=false) for cls in classes_]))
    data.X = X[:, idx]
    data.y = y[:, idx]
    data.output_encoder.y = data.output_encoder.y[idx]

    return data

end