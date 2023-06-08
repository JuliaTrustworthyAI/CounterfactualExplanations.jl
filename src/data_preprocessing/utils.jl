"""
    _subset(data::CounterfactualData, idx::Vector{Int})

Creates a subset of the `data`.
"""
function _subset(data::CounterfactualData, idx::Vector{Int})
    dsub = deepcopy(data)
    dsub.X = dsub.X[:, idx]
    dsub.y = dsub.y[:, idx]
    dsub.output_encoder.y = data.output_encoder.y[idx]
    dsub.output_encoder.labels = data.output_encoder.labels[idx]
    return dsub
end

"""
    train_test_split(data::CounterfactualData;test_size=0.2)

Splits data into train and test split where `test_size` is the proportion of the data to be used for testing.
"""
function train_test_split(data::CounterfactualData; test_size=0.2)
    N = size(data.X, 2)
    classes_ = data.y_levels
    n_per_class = round(N / length(classes_))
    y = data.output_encoder.y
    test_idx = sort(
        reduce(
            vcat,
            [
                sample(
                    findall(vec(y .== cls)),
                    Int(floor(test_size * n_per_class));
                    replace=false,
                ) for cls in classes_
            ],
        ),
    )
    train_idx = setdiff(1:N, test_idx)
    train_data = _subset(data, train_idx)
    test_data = _subset(data, test_idx)
    return train_data, test_data
end

"""
    subsample(data::CounterfactualData, n::Int)

Helper function to randomly subsample `data::CounterfactualData`.
"""
function subsample(data::CounterfactualData, n::Int)
    X, y = data.X, data.output_encoder.y
    classes_ = data.y_levels
    n_classes = length(classes_)
    n_per_class = Int(round(n / n_classes))
    y_cls = data.output_encoder.labels

    idx = sort(
        reduce(
            vcat,
            [
                sample(findall(vec(y_cls .== cls)), n_per_class; replace=true) for
                cls in classes_
            ],
        ),
    )
    X = X[:, idx]
    y = y[idx]
    new_data = CounterfactualData(
        X,
        y;
        domain=data.domain,
        features_continuous=data.features_continuous,
        features_categorical=data.features_categorical,
        mutability=data.mutability,
        standardize=data.standardize,
    )

    return new_data
end

"""
    preprocess_data_for_mlj(data::CounterfactualData)

Helper function to preprocess `data::CounterfactualData` for MLJ models.

# Arguments
- `data::CounterfactualData`: The data to be preprocessed.

# Returns
- (`df_x`, `y`): A tuple containing the preprocessed data, with `df_x` being a DataFrame object and `y` being a categorical vector.

# Example
X, y = preprocess_data_for_mlj(data)
"""
function preprocess_data_for_mlj(data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)

    X = Float32.(X)
    y = Float32.(y)[1, :]

    df_x = DataFrame(X', :auto)
    y = categorical(y)
    return df_x, y
end
