using CategoricalArrays

# Constants:
"""
    RawTargetType

A type union for the allowed types for the `target` variable.
"""
const RawTargetType = Union{Int,AbstractFloat,String,Symbol}

"""
    EncodedTargetType

Type of encoded target variable.
"""
const EncodedTargetType = Real

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

"""
    OutputEncoder

The `OutputEncoder` takes a raw output array (`y`) and encodes it.
"""
struct OutputEncoder
    y::RawOutputArrayType
end

"""
    (encoder::OutputEncoder)(ynew::Union{Nothing,RawTargetType}=nothing)

The `OutputEncoder` can be called on `nothing` or a single element `ynew::RawTargetType`. In the first case, the encoding is applied the the yaw output array (`y`). In the second case, the encoding is applied to `ynew`.
"""
function (encoder::OutputEncoder)(ynew::Union{Nothing,RawTargetType}=nothing)

    # Setup:
    y = encoder.y
    y_levels = levels(y)
    if !isnothing(ynew)
        ynew = get_target_index(y_levels, ynew)
    end

    # Transformations:
    if typeof(y) <: CategoricalArray
        y_cat = y
        y = permutedims(Int.(y_cat.refs))
        # Binary case:
        if length(levels(y_cat)) == 2
            y = y .- 1
            if !isnothing(ynew)
                ynew -= 1
            end
        end
        y_levels = levels(y_cat)
    elseif typeof(y) <: AbstractVector
        y = permutedims(y)
    end

    # Output:
    if isnothing(ynew)
        return y
    else
        return ynew
    end

end

"""
    get_target_index(y_levels, target)

Utility that returns the index of `target` in `y_levels`.
"""
function get_target_index(y_levels, target)
    @assert target in y_levels "Specified `target` variable does not match any values of `y`."
    findall(y_levels .== target)[1]
end