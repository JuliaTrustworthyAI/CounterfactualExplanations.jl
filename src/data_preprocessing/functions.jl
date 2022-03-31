using Flux

struct CounterfactualData
    X::AbstractArray
    y::AbstractArray
    mutability::Union{Vector{Symbol},Nothing}
    categorical::Union{Vector{Int},Nothing}
    continuous::Union{Vector{Int},Nothing}
    bounds_continuous::Union{Any,Nothing}
    CounterfactualData(X,y,mutability,categorical,continuous,bounds_continuous) = length(size(X)) != 2 ? error("Data should be in tabular format") : new(X,y,mutability,categorical,continuous,bounds_continuous)
end

"""
    CounterfactualData(X::AbstractArray, y::AbstractArray)

This function prepares features `X` and labels `y` to be used with the package.

# Examples

```julia-repl
using CounterfactualExplanations.Data
x, y = toy_data_linear()
X = hcat(x...)
counterfactual_data = CounterfactualData(X,y')
```

"""
function CounterfactualData(
    X::AbstractArray, y::AbstractArray;
    mutability::Union{Vector{Symbol},Nothing}=nothing,
    categorical::Union{Vector{Int},Nothing}=nothing,
    continuous::Union{Vector{Int},Nothing}=nothing,
    bounds_continuous::Union{Any,Nothing}=nothing
)

    # If nothing supplied, assume all continuous:
    if isnothing(categorical) && isnothing(continuous)
        continuous = 1:size(X)[1]
    end

    # If tuple is supplied, assume it counts for all continuous variables:
    if typeof(bounds_continuous) <: Tuple
        bounds_continuous = [bounds_continuous for i in 1:size(X)[1]]
    end

    counterfactual_data = CounterfactualData(X, y, mutability, categorical, continuous, bounds_continuous)

    return counterfactual_data
end

select_factual(counterfactual_data::CounterfactualData, index::Int) = counterfactual_data.X[:,index]

mutability_constraints(counterfactual_data::CounterfactualData) = isnothing(counterfactual_data.mutability) ? [:both for i in 1:size(counterfactual_data.X)[1]] : counterfactual_data.mutability

function apply_domain_constraints(counterfactual_data::CounterfactualData, x::AbstractArray) 
    
    # Continuous variables:
    for i in counterfactual_data.continuous
        x[i] = clamp(x[i], counterfactual_data.bounds_continuous[i][1], counterfactual_data.bounds_continuous[i][2])
    end

    return x
    
end