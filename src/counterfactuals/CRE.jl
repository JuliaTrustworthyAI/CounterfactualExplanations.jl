"""
    Rule

A `Rule` is just a list of bounds for the different features. See also [`CRE`](@ref).
"""
struct Rule
    bounds::Vector{Tuple}
end

"""
    CRE <: AbstractCounterfactualExplanation

A Counterfactual Rule Explanation (CRE) is a global explanation for a given `target`, model `M`, `data` and `generator`.
"""
struct CRE <: AbstractCounterfactualExplanation
    target::RawTargetType
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractModel
    generator::Generators.AbstractGenerator
    rules::Vector{Rule}
    meta_rules::Union{Nothing,Vector{Rule}}
    other::Union{Dict,Nothing}
end

"""
    (cre::CRE)(x::AbstractArray)

Generates a local counterfactual point explanation for `x` using the `generator`.
"""
function (cre::CRE)(x::AbstractArray)
    return cre(x::AbstractArray, cre.generator)
end
