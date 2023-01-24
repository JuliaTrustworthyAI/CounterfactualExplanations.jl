using CategoricalArrays

# Constants:
"""
    TargetType

A type union for the allowed types for the `target` variable.
"""
const TargetType = Union{Int,AbstractFloat,String,Symbol}

"""
    OutputArrayType 

A type union for the allowed type for the output array `y`.
"""
const OutputArrayType = Union{AbstractVector,AbstractMatrix,CategoricalVector}

# Utilities:
function get_target_index(y_levels, target)
    @assert target in y_levels "Specified `target` variable does not match any values of `y`."
    findall(y_levels .== target)[1]
end