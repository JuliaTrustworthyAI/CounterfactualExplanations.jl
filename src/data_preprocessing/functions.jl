using Flux
using MultivariateStats
using StatsBase
using UMAP

mutable struct CounterfactualData
    X::AbstractMatrix
    y::AbstractMatrix
    mutability::Union{Vector{Symbol},Nothing}
    domain::Union{Any,Nothing}
    features_categorical::Union{Vector{Int},Nothing}
    features_continuous::Union{Vector{Int},Nothing}
    standardize::Bool
    dt::Union{Nothing,StatsBase.AbstractDataTransform}
    compressor::Union{Nothing,MultivariateStats.PCA,UMAP.UMAP_}
    generative_model::Union{Nothing, GenerativeModels.AbstractGenerativeModel} # generative model
    function CounterfactualData(X,y,mutability,domain,features_categorical,features_continuous,standardize,dt,compressor,generative_model)
        conditions = []
        conditions = vcat(
            conditions...,
            length(size(X)) != 2 ? error("Data should be in tabular format") : true,
        )
        conditions = vcat(
            conditions...,
            size(X)[2] != size(y)[2] ?
            throw(
                DimensionMismatch(
                    "Number of output observations is $(size(y)[2]). Expected: $(size(X)[2])",
                ),
            ) : true,
        )
        if all(conditions)
            new(X,y,mutability,domain,features_categorical,features_continuous,standardize,dt,compressor,generative_model)
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
    X::AbstractMatrix, y::AbstractMatrix;
    mutability::Union{Vector{Symbol},Nothing}=nothing,
    domain::Union{Any,Nothing}=nothing,
    features_categorical::Union{Vector{Int},Nothing}=nothing,
    features_continuous::Union{Vector{Int},Nothing}=nothing,
    standardize::Bool=false,
    generative_model::Union{Nothing, GenerativeModels.AbstractGenerativeModel}=nothing
)

    # If nothing supplied, assume all continuous:
    if isnothing(features_categorical) && isnothing(features_continuous)
        features_continuous = 1:size(X, 1)
    end

    # Defaults:
    compressor = nothing                                                                # dimensionality reduction
    domain = typeof(domain) <: Tuple ? [domain for var in features_continuous] : domain          # domain constraints

    # Data transformations:
    dt = fit(ZScoreTransform, X[features_continuous,:], dims=2)        # standardization

    counterfactual_data = CounterfactualData(
        X, y, mutability, domain, features_categorical, features_continuous, 
        standardize, dt, compressor, generative_model
    )

    return counterfactual_data
end

"""
    select_factual(counterfactual_data::CounterfactualData, index::Int)

A convenience method that can be used to access the the feature matrix.
"""
select_factual(counterfactual_data::CounterfactualData, index::Int) =
    reshape(collect(selectdim(counterfactual_data.X, 2, index)), :, 1)
select_factual(
    counterfactual_data::CounterfactualData,
    index::Union{Vector{Int},UnitRange{Int}},
) = zip([select_factual(counterfactual_data, i) for i in index])

"""
    mutability_constraints(counterfactual_data::CounterfactualData)

A convience function that returns the mutability constraints. If none were specified, it is assumed that all features are mutable in `:both` directions.
"""
mutability_constraints(counterfactual_data::CounterfactualData) =
    isnothing(counterfactual_data.mutability) ?
    [:both for i = 1:size(counterfactual_data.X)[1]] : counterfactual_data.mutability

"""
    apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray) 

A subroutine that is used to apply the predetermined domain constraints.
"""
function apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray)

    # Continuous variables:
    if !isnothing(counterfactual_data.domain)
        for i in counterfactual_data.features_continuous
            x[i] = clamp(x[i], counterfactual_data.domain[i][1], counterfactual_data.domain[i][2])
        end
    end

    return x

end

"""
    input_dim(counterfactual_data::CounterfactualData)

Helper function that returns the input dimension (number of features) of the data. 

"""
input_dim(counterfactual_data::CounterfactualData) = size(counterfactual_data.X)[1]

"""
    unpack(data::CounterfactualData)

Helper function that unpacks data.
"""
function unpack(data::CounterfactualData)
    return data.X, data.y
end


"""
    has_pretrained_generative_model(counterfactual_data::CounterfactualData)

Checks if generative model is present and trained.
"""
has_pretrained_generative_model(counterfactual_data::CounterfactualData) =
    !isnothing(counterfactual_data.generative_model) &&
    counterfactual_data.generative_model.trained


"""
    get_generative_model(counterfactual_data::CounterfactualData)

Returns the underlying generative model. If there is no existing model available, the default generative model (VAE) is used. Otherwise it is expected that existing generative model has been pre-trained or else a warning is triggered.
"""
function get_generative_model(counterfactual_data::CounterfactualData; kwargs...)
    if !has_pretrained_generative_model(counterfactual_data)
        @info "No pre-trained generative model found. Using default generative model. Begin training."
        counterfactual_data.generative_model =
            GenerativeModels.VAE(input_dim(counterfactual_data); kwargs...)
        X = counterfactual_data.X
        y = counterfactual_data.y
        GenerativeModels.train!(counterfactual_data.generative_model, X, y)
        @info "Training of generative model completed."
    else
        if !counterfactual_data.generative_model.trained
            @warn "The provided generative model has not been trained. Latent space search is likely to perform poorly."
        end
    end
    return counterfactual_data.generative_model
end
