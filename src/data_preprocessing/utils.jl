using CategoricalArrays: CategoricalArrays
using StatsBase: sample

"Treat `CounterfactualData` as scalar when broadcasting."
Base.broadcastable(data::CounterfactualData) = Ref(data)

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
    train_test_split(data::CounterfactualData;test_size=0.2,keep_class_ratio=false)

Splits data into train and test split.

# Arguments
- `data::CounterfactualData`: The data to be preprocessed.
- `test_size=0.2`: Proportion of the data to be used for testing. 
- `keep_class_ratio=false`: Decides whether to sample equally from each class, or keep their relative size.

# Returns
- (`train_data::CounterfactualData`, `test_data::CounterfactualData`): A tuple containing the train and test splits.

# Example
train, test = train_test_split(data, test_size=0.1, keep_class_ratio=true)
"""
function train_test_split(data::CounterfactualData; test_size=0.2, keep_class_ratio=false)
    N = size(data.X, 2)
    classes_ = data.y_levels
    y = data.output_encoder.y
    n_per_class = round(N / length(classes_))

    if keep_class_ratio
        class_ratios = [length(findall(vec(y .== cls))) / length(y) for cls in classes_]
        class_samples = [
            sample(
                findall(vec(y .== cls)),
                Int(floor(test_size * cls_ratio * N));
                replace=false,
            ) for (cls, cls_ratio) in zip(classes_, class_ratios)
        ]
    else
        class_samples = [
            sample(
                findall(vec(y .== cls)), Int(floor(test_size * n_per_class)); replace=false
            ) for cls in classes_
        ]
    end

    test_idx = sort(reduce(vcat, class_samples))
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
        input_encoder=data.input_encoder,
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
    y = convert_to_1d(y, data.y_levels)

    df_x = DataFrames.DataFrame(X', :auto)
    y = CategoricalArrays.categorical(y)
    return df_x, y
end

"""
    convert_to_1d(y::Matrix, y_levels::AbstractArray)

Helper function to convert a one-hot encoded matrix to a vector of labels.
This is necessary because MLJ models require the labels to be represented as a vector,
but the synthetic datasets in this package hold the labels in one-hot encoded form.

# Arguments
- `y::Matrix`: The one-hot encoded matrix.
- `y_levels::AbstractArray`: The levels of the categorical variable.

# Returns
- `labels`: A vector of labels.
"""
function convert_to_1d(y::Matrix, y_levels::AbstractArray)
    # Number of rows in the onehot_encoded matrix corresponds to the number of data points
    num_data_points = size(y, 2)

    # Initialize an empty vector to hold the labels
    labels = Vector{Int}(undef, num_data_points)

    # For each data point
    for i in 1:num_data_points
        # Find the index of the column with a 1
        index = findfirst(y[:, i] .== 1)

        # Use this index to get the corresponding label from levels
        labels[i] = y_levels[index]
    end

    return labels
end

"""
    input_dim(counterfactual_data::CounterfactualData)

Helper function that returns the input dimension (number of features) of the data. 

"""
input_dim(counterfactual_data::CounterfactualData) = size(counterfactual_data.X)[1]

"""
    unpack_data(data::CounterfactualData)

Helper function that unpacks data.
"""
function unpack_data(data::CounterfactualData)
    return data.X, data.y
end

"""
    select_factual(counterfactual_data::CounterfactualData, index::Int)

A convenience method that can be used to access the feature matrix.
"""
function select_factual(counterfactual_data::CounterfactualData, index::Int)
    return reshape(collect(selectdim(counterfactual_data.X, 2, index)), :, 1)
end

"""
    select_factual(counterfactual_data::CounterfactualData, index::Union{Vector{Int},UnitRange{Int}})

A convenience method that can be used to access the feature matrix.
"""
function select_factual(
    counterfactual_data::CounterfactualData, index::Union{Vector{Int},UnitRange{Int}}
)
    return zip([select_factual(counterfactual_data, i) for i in index])
end

"""
    outdim(data::CounterfactualData)

Returns the number of output classes.
"""
function outdim(data::CounterfactualData)
    return length(data.y_levels)
end
