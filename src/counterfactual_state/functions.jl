################################################################################
# --------------- Constructor for counterfactual state:
################################################################################
struct State
    x::AbstractArray
    s′::AbstractArray
    f::Function
    target_encoded::Union{Number, AbstractVector}
    M::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

using Flux
function guess_loss(counterfactual_state::State)
    if :type in fieldnames(typeof(counterfactual_state.M))
        if counterfactual_state.M == :classification_binary
            loss_fun = Flux.Losses.logitbinarycrossentropy
        elseif counterfactual_state.M == :classification_multi
            loss_fun = Flux.Losses.logitcrossentropy
        else
            loss_fun = Flux.Losses.mse
        end
    else
        loss_fun = nothing
    end
    return loss_fun
end