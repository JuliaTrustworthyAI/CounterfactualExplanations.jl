"""
    CounterfactualData(
        X::AbstractMatrix, y::AbstractMatrix;
        mutability::Union{Vector{Symbol},Nothing}=nothing,
        domain::Union{Any,Nothing}=nothing,
        features_categorical::Union{Vector{Int},Nothing}=nothing,
        features_continuous::Union{Vector{Int},Nothing}=nothing,
        standardize::Bool=false
    )
Stores data and metadata for counterfactual explanations.
"""

mutable struct CounterfactualData
    X::AbstractMatrix
    y::EncodedOutputArrayType
    likelihood::Symbol
    mutability::Union{Vector{Symbol},Nothing}
    domain::Union{Any,Nothing}
    features_categorical::Union{Vector{Vector{Int}},Nothing}
    features_continuous::Union{Vector{Int},Nothing}
    standardize::Bool
    dt::Union{
        Nothing,
        StatsBase.AbstractDataTransform,
        MultivariateStats.AbstractDimensionalityReduction,
    }
    compressor::Union{Nothing,MultivariateStats.PCA}
    generative_model::Union{Nothing,GenerativeModels.AbstractGenerativeModel} # generative model
    y_levels::AbstractVector
    output_encoder::OutputEncoder
    function CounterfactualData(
        X,
        y,
        likelihood,
        mutability,
        domain,
        features_categorical,
        features_continuous,
        standardize,
        dt,
        compressor,
        generative_model,
        y_levels,
        output_encoder,
    )

        # Conditions:
        conditions = []
        # Feature dimension:
        conditions = vcat(
            conditions...,
            length(size(X)) != 2 ? error("Data should be in tabular format") : true,
        )
        # Output dimension:
        conditions = vcat(
            conditions...,
            if size(X)[2] != size(y)[2]
                throw(
                    DimensionMismatch(
                        "Number of output observations is $(size(y)[2]). Expected it to match the number of input observations: $(size(X)[2]).",
                    ),
                )
            else
                true
            end,
        )
        # Likelihood:
        available_likelihoods = [:classification_binary, :classification_multi]
        @assert likelihood ∈ available_likelihoods "Specified likelihood not available. Needs to be one of: $(available_likelihoods)."

        if all(conditions)
            new(
                X,
                y,
                likelihood,
                mutability,
                domain,
                features_categorical,
                features_continuous,
                standardize,
                dt,
                compressor,
                generative_model,
                y_levels,
                output_encoder,
            )
        end
    end
end

"""
    CounterfactualData(
        X::AbstractMatrix, y::AbstractMatrix;
        mutability::Union{Vector{Symbol},Nothing}=nothing,
        domain::Union{Any,Nothing}=nothing,
        features_categorical::Union{Vector{Int},Nothing}=nothing,
        features_continuous::Union{Vector{Int},Nothing}=nothing,
        standardize::Bool=false
    )

This outer constructor method prepares features `X` and labels `y` to be used with the package. Mutability and domain constraints can be added for the features. The function also accepts arguments that specify which features are categorical and which are continues. These arguments are currently not used. 

# Examples

```julia-repl
using CounterfactualExplanations.Data
x, y = toy_data_linear()
X = hcat(x...)
counterfactual_data = CounterfactualData(X,y')
```

"""
function CounterfactualData(
    X::AbstractMatrix,
    y::RawOutputArrayType;
    mutability::Union{Vector{Symbol},Nothing}=nothing,
    domain::Union{Any,Nothing}=nothing,
    features_categorical::Union{Vector{Vector{Int}},Nothing}=nothing,
    features_continuous::Union{Vector{Int},Nothing}=nothing,
    standardize::Bool=false,
    generative_model::Union{Nothing,GenerativeModels.AbstractGenerativeModel}=nothing,
)

    # Output variable:
    y_raw = deepcopy(y)
    output_encoder = OutputEncoder(y_raw, nothing)
    y, y_levels, likelihood = output_encoder()

    # Feature type indices:
    if isnothing(features_categorical) && isnothing(features_continuous)
        features_continuous = 1:size(X, 1)
    elseif !isnothing(features_categorical) && isnothing(features_continuous)
        features_all = 1:size(X, 1)
        cat_indices = reduce(vcat, features_categorical)
        features_continuous = findall(map(i -> !(i ∈ cat_indices), features_all))
    end

    # Defaults:
    compressor = nothing                                                                # dimensionality reduction
    domain = typeof(domain) <: Tuple ? [domain for var in features_continuous] : domain          # domain constraints

    counterfactual_data = CounterfactualData(
        X,
        y,
        likelihood,
        mutability,
        domain,
        features_categorical,
        features_continuous,
        standardize,
        nothing,
        compressor,
        generative_model,
        y_levels,
        output_encoder,
    )

    # Data transformations:
    if transformable_features(counterfactual_data) !=
        counterfactual_data.features_continuous
        @warn "Some of the underlying features are constant."
    end
    counterfactual_data.dt = StatsBase.fit(
        StatsBase.ZScoreTransform,
        X[transformable_features(counterfactual_data), :];
        dims=ndims(X),
    )        # standardization

    return counterfactual_data
end

"""
    function CounterfactualData(
        X::Tables.MatrixTable,
        y::RawOutputArrayType;
        kwrgs...
    )
    
Outer constructor method that accepts a `Tables.MatrixTable`. By default, the indices of categorical and continuous features are automatically inferred the features' `scitype`.

"""
function CounterfactualData(X::Tables.MatrixTable, y::RawOutputArrayType; kwrgs...)
    features_categorical = findall([scitype(x) <: AbstractVector{<:Finite} for x in X])
    features_categorical =
        length(features_categorical) == 0 ? nothing : features_categorical
    features_continuous = findall([scitype(x) <: AbstractVector{<:Continuous} for x in X])
    features_continuous = length(features_continuous) == 0 ? nothing : features_continuous
    X = permutedims(MLJBase.matrix(X))

    counterfactual_data = CounterfactualData(X, y; kwrgs...)

    return counterfactual_data
end

"""
    reconstruct_cat_encoding(counterfactual_data::CounterfactualData, x::Vector)

Reconstruct the categorical encoding for a single instance. 
"""
function reconstruct_cat_encoding(counterfactual_data::CounterfactualData, x::AbstractArray)
    features_categorical = counterfactual_data.features_categorical

    if isnothing(features_categorical)
        return x
    end

    x = vec(x)
    map(features_categorical) do cat_group_index
        if length(cat_group_index) > 1
            x[cat_group_index] = Int.(x[cat_group_index] .== maximum(x[cat_group_index]))
            if sum(x[cat_group_index]) > 1
                ties = findall(x[cat_group_index] .== 1)
                _x = zeros(length(x[cat_group_index]))
                winner = rand(ties, 1)[1]
                _x[winner] = 1
                x[cat_group_index] = _x
            end
        else
            x[cat_group_index] = [round(clamp(x[cat_group_index][1], 0, 1))]
        end
    end

    return x
end

"""
    transformable_features(counterfactual_data::CounterfactualData)

Dispatches the `transformable_features` function to the appropriate method based on the type of the `dt` field.
"""
function transformable_features(counterfactual_data::CounterfactualData)
    return transformable_features(counterfactual_data, counterfactual_data.dt)
end

"""
    transformable_features(counterfactual_data::CounterfactualData, dt::Any)

By default, all continuous features are transformable. This function returns the indices of all continuous features.
"""
function transformable_features(counterfactual_data::CounterfactualData, dt::Any)
    return counterfactual_data.features_continuous
end

"""
    transformable_features(counterfactual_data::CounterfactualData, dt::ZScoreTransform)

Returns the indices of all continuous features that can be transformed. For constant features `ZScoreTransform` returns `NaN`.
"""
function transformable_features(
    counterfactual_data::CounterfactualData, dt::ZScoreTransform
)
    # Find all columns that have varying values:
    idx_not_all_equal = [
        length(unique(counterfactual_data.X[i, :])) != 1 for
        i in counterfactual_data.features_continuous
    ]
    # Returns indices of columns that have varying values:
    return counterfactual_data.features_continuous[idx_not_all_equal]
end
