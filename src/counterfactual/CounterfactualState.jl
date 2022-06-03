module CounterfactualState

using ..Models

################################################################################
# --------------- Constructor for counterfactual state:
################################################################################
struct State
    s′::AbstractArray
    f::Function
    target_encoded::Union{Number, AbstractVector}
    M::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

end
