using Flux

"""
Constructor for counterfactual state.
"""
struct State
    x::AbstractArray
    s′::AbstractArray
    f::Function
    y′::AbstractArray
    target::Number
    target_encoded::Union{Number, AbstractArray, Nothing}
    γ::Number
    threshold_reached::Bool
    M::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

"""
    guess_loss(counterfactual_state::State)

Guesses the loss function to be used for the counterfactual search in case `likelihood` field is specified for the [`AbstractFittedModel`](@ref) instance and no loss function was explicitly declared for [`AbstractGenerator`](@ref) instance.
"""
function guess_loss(counterfactual_state::State)
    if :likelihood in fieldnames(typeof(counterfactual_state.M))
        if counterfactual_state.M.likelihood == :classification_binary
            loss_fun = Flux.Losses.logitbinarycrossentropy
        elseif counterfactual_state.M.likelihood == :classification_multi
            loss_fun = Flux.Losses.logitcrossentropy
        else
            loss_fun = Flux.Losses.mse
        end
    else
        loss_fun = nothing
    end
    return loss_fun
end