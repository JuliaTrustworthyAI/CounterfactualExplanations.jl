
function generate_counterfactual(
    x::Vector{AbstractFloat},
    target::RawTargetType,
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::GrowingSpheresGenerator;
    num_counterfactuals::Int=1,
    max_iter::Int=100,
)
    counterfactuals = generate(x, generator)
    return counterfactuals
end

mutable struct GrowingSpheresCounterfactualExplanation <: AbstractCounterfactualExplanation
    x′::Vector{Vector{AbstractFloat}}
end
