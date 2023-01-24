using CategoricalArrays

# Constants:
"""
    TargetType

A type union for the allowed types for the `target` variable.
"""
const TargetType = Union{Int,AbstractFloat,String,Symbol}

"""
    RawOutputArrayType 

A type union for the allowed type for the output array `y`.
"""
const RawOutputArrayType = Union{AbstractVector,AbstractMatrix,CategoricalVector}

"""
    EncodedOutputArrayType

Type of encoded output array.
"""
const EncodedOutputArrayType = AbstractMatrix

# Utilities:
function encode_output(y::RawOutputArrayType)
    y_levels = levels(y)
    if typeof(y) <: CategoricalArray
        y_cat = y
        y = permutedims(Int.(y_cat.refs))
        if length(levels(y_cat)) == 2
            # Binary case:
            y = y .- 1
        end
        y_levels = levels(y_cat)
    elseif typeof(y) <: AbstractVector
        y = permutedims(y)
    end
    return y, y_levels
end

"""
    get_target_index(y_levels, target)

Utility that returns the index of `target` in `y_levels`.
"""
function get_target_index(y_levels, target)
    @assert target in y_levels "Specified `target` variable does not match any values of `y`."
    findall(y_levels .== target)[1]
end