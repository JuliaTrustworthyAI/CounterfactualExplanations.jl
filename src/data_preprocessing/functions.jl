using Flux

struct CounterfactualData
    X::AbstractArray
    y::AbstractArray
    mutability::Union{Vector{Symbol},Nothing}
    categorical::Union{Vector{Int64},Nothing}
    continuous::Union{Vector{Int64},Nothing}
    bounds_continuous::Union{Vector{Tuple{Number,Number}},Tuple{Number,Number},Nothing}
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
CounterfactualData(X::AbstractArray, y::AbstractArray) = CounterfactualData(X, y, nothing, nothing, nothing, nothing)

select_factual(cfd::CounterfactualData, index::Int) = cfd.X[:,index]

mutability_constraints(cfd::CounterfactualData) = isnothing(cfd.mutability) ? [:both for i in 1:size(cfd.X)[1]] : cfd.mutability