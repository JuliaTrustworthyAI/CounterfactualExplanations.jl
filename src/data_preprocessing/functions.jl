using Flux
using StatsBase

struct CounterfactualData
    X::AbstractMatrix
    y::AbstractMatrix
    mutability::Union{Vector{Symbol},Nothing}
    domain::Union{Any,Nothing}
    categorical::Union{Vector{Int},Nothing}
    continuous::Union{Vector{Int},Nothing}
    standardize::Bool
    dt::StatsBase.AbstractDataTransform
    function CounterfactualData(X,y,mutability,domain,categorical,continuous,standardize,dt)
        conditions = []
        conditions = vcat(conditions..., length(size(X)) != 2 ? error("Data should be in tabular format") : true)
        conditions = vcat(conditions..., size(X)[2] != size(y)[2] ? throw(DimensionMismatch("Number of output observations is $(size(y)[2]). Expected: $(size(X)[2])")) : true)
        if all(conditions)
            new(X,y,mutability,domain,categorical,continuous,standardize,dt)
        end
    end
end

"""
    CounterfactualData(
        X::AbstractMatrix, y::AbstractMatrix;
        mutability::Union{Vector{Symbol},Nothing}=nothing,
        domain::Union{Any,Nothing}=nothing,
        categorical::Union{Vector{Int},Nothing}=nothing,
        continuous::Union{Vector{Int},Nothing}=nothing,
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
    categorical::Union{Vector{Int},Nothing}=nothing,
    continuous::Union{Vector{Int},Nothing}=nothing,
    standardize::Bool=false
)

    # If nothing supplied, assume all continuous:
    if isnothing(categorical) && isnothing(continuous)
        continuous = 1:size(X)[1]
    end

    # If tuple is supplied, assume it counts for all continuous variables:
    if typeof(domain) <: Tuple
        domain = [domain for i in 1:size(X)[1]]
    end

    # Data transformer:
    dt = fit(ZScoreTransform, X, dims=2)

    counterfactual_data = CounterfactualData(X, y, mutability, domain, categorical, continuous, standardize, dt)

    return counterfactual_data
end

"""
    select_factual(counterfactual_data::CounterfactualData, index::Int)

A convenience method that can be used to access the the feature matrix.
"""
select_factual(counterfactual_data::CounterfactualData, index::Int) = counterfactual_data.X[:,index]

"""
    mutability_constraints(counterfactual_data::CounterfactualData)

A convience function that returns the mutability constraints. If none were specified, it is assumed that all features are mutable in `:both` directions.
"""
mutability_constraints(counterfactual_data::CounterfactualData) = isnothing(counterfactual_data.mutability) ? [:both for i in 1:size(counterfactual_data.X)[1]] : counterfactual_data.mutability

"""
    apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray) 

A subroutine that is used to apply the predetermined domain constraints.
"""
function apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray) 
    
    # Continuous variables:
    if !isnothing(counterfactual_data.domain)
        for i in counterfactual_data.continuous
            x[i] = clamp(x[i], counterfactual_data.domain[i][1], counterfactual_data.domain[i][2])
        end
    end

    return x
    
end