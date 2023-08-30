"The `generate_counterfactual` method is parallelizable."
ProcessStyle(::Type{<:typeof(generate_counterfactual)}) = IsParallel()

"The `evaluate` function is parallelizable."
function ProcessStyle(::Type{<:typeof(CounterfactualExplanations.Evaluation.evaluate)})
    return IsParallel()
end
