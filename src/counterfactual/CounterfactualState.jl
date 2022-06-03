################################################################################
# --------------- Constructor for counterfactual state:
################################################################################
struct CounterfactualState
    x′::AbstractArray
    target_encoded::Union{Number, AbstractVector}
    M::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

