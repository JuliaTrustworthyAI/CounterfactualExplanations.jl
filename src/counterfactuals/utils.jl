"""
    function terminated(ce::CounterfactualExplanation)

A convenience method that checks if the counterfactual search has terminated.
"""
function terminated(ce::CounterfactualExplanation)
    return Convergence.converged(ce.convergence, ce) || steps_exhausted(ce)
end

"""
    steps_exhausted(ce::CounterfactualExplanation) 

A convenience method that checks if the number of maximum iterations has been exhausted.
"""
function steps_exhausted(ce::CounterfactualExplanation)
    return ce.search[:iteration_count] == ce.convergence.max_iter
end

"""
    total_steps(ce::CounterfactualExplanation)

A convenience method that returns the total number of steps of the counterfactual search.
"""
function total_steps(ce::CounterfactualExplanation)
    return ce.search[:iteration_count]
end

"""
    in_target_class(ce::CounterfactualExplanation)

Check if the counterfactual is in the target class.
"""
function in_target_class(ce::AbstractCounterfactualExplanation)
    return Models.predict_label(ce.M, ce.data, decode_state(ce))[1] == ce.target
end

"""
    output_dim(ce::CounterfactualExplanation)

A convenience method that returns the output dimension of the predictive model.
"""
function output_dim(ce::CounterfactualExplanation)
    return size(Models.probs(ce.M, ce.x))[1]
end

"""
    guess_loss(ce::CounterfactualExplanation)

Guesses the loss function to be used for the counterfactual search in case `likelihood` field is specified for the [`AbstractFittedModel`](@ref) instance and no loss function was explicitly declared for [`AbstractGenerator`](@ref) instance.
"""
function guess_loss(ce::CounterfactualExplanation)
    if :likelihood in fieldnames(typeof(ce.M))
        if ce.M.likelihood == :classification_binary
            loss_fun = Objectives.logitbinarycrossentropy
        elseif ce.M.likelihood == :classification_multi
            loss_fun = Objectives.logitcrossentropy
        else
            loss_fun = Flux.Losses.mse
        end
    else
        loss_fun = nothing
    end
    return loss_fun
end

"""
    get_meta(ce::CounterfactualExplanation)

Returns meta data for a counterfactual explanation.
"""
function get_meta(ce::CounterfactualExplanation)
    meta_data = Dict(:model => Symbol(ce.M), :generator => Symbol(ce.generator))
    return meta_data
end

"""
    adjust_shape(
        ce::CounterfactualExplanation, 
        x::AbstractArray
    )

A convenience method that adjusts the dimensions of `x`.
"""
function adjust_shape(ce::CounterfactualExplanation, x::AbstractArray)
    s′ = repeat(x; outer=(1, ce.num_counterfactuals))
    return s′
end

"""
    adjust_shape!(ce::CounterfactualExplanation)

A convenience method that adjusts the dimensions of the counterfactual state and related fields.
"""
function adjust_shape!(ce::CounterfactualExplanation)

    # Dimensionality:
    x = deepcopy(ce.x)
    s′ = adjust_shape(ce, x)      # augment to account for specified number of counterfactuals
    ce.s′ = s′
    target_encoded = ce.target_encoded
    ce.target_encoded = adjust_shape(ce, target_encoded)

    # Parameters:
    params = ce.params
    params[:mutability] = adjust_shape(ce, params[:mutability])      # augment to account for specified number of counterfactuals
    return ce.params = params
end
