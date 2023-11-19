mutable struct GrowingSpheresCounterfactualExplanation <: AbstractCounterfactualExplanation
    x::AbstractArray
    target::RawTargetType
    s′::AbstractArray
    x′::AbstractArray
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractFittedModel
    generator::Generators.GrowingSpheresGenerator
    num_counterfactuals::Int
    max_iter::Int
    search::Dict
end