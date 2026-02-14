"""
    outdim(ce::CounterfactualExplanation)

A convenience method that returns the output dimension of the predictive model.
"""
function outdim(ce::CounterfactualExplanation)
    return CounterfactualExplanations.DataPreprocessing.outdim(ce.data)
end

"""
    guess_loss(ce::CounterfactualExplanation)

Guesses the loss function to be used for the counterfactual search in case `likelihood` field is specified for the [`AbstractModel`](@ref) instance and no loss function was explicitly declared for [`AbstractGenerator`](@ref) instance.
"""
function guess_loss(ce::CounterfactualExplanation)
    M = ce.M
    if :likelihood in fieldnames(typeof(M))
        if M.likelihood == :classification_binary
            loss_fun = Objectives.logitbinarycrossentropy
        elseif M.likelihood == :classification_multi
            loss_fun = Objectives.logitcrossentropy
        else
            loss_fun = Objectives.mse
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
    counterfactual_state = repeat(x; outer=(1, ce.num_counterfactuals))
    return counterfactual_state
end

"""
    adjust_shape!(ce::CounterfactualExplanation)

A convenience method that adjusts the dimensions of the counterfactual state and related fields.
"""
function adjust_shape!(ce::CounterfactualExplanation)

    # Dimensionality:
    counterfactual_state = adjust_shape(ce, ce.counterfactual)      # augment to account for specified number of counterfactuals
    ce.counterfactual_state = counterfactual_state
    target_encoded = ce.target_encoded
    ce.target_encoded = adjust_shape(ce, target_encoded)

    search = ce.search
    search[:mutability] = adjust_shape(ce, search[:mutability])      # augment to account for specified number of counterfactuals
    ce.search = search

    return ce
end

"""
    find_potential_neighbours(
        ce::AbstractCounterfactualExplanation, data::CounterfactualData, n::Int=1000
    )

Finds potential neighbors for the selected factual data point.
"""
function find_potential_neighbours(
    ce::AbstractCounterfactualExplanation, data::CounterfactualData, n::Int=1000
)
    # Get IDs of samples in target class (ground truth):
    ids = findall(data.output_encoder.labels .== ce.target)
    n_candidates = minimum([size(data.y, 2), n])
    candidates = DataPreprocessing.select_factual(data, rand(ids, n_candidates))
    potential_neighbours = reduce(hcat, map(x -> x[1], collect(candidates)))
    return potential_neighbours
end

"""
    find_potential_neighbours(ce::CounterfactualExplanation, n::Int=1000)

Overloads the function for [`CounterfactualExplanation`](@ref) to use the counterfactual data's labels if no data is provided.
"""
function find_potential_neighbours(ce::CounterfactualExplanation, n::Int=1000)
    return find_potential_neighbours(ce, ce.data, n)
end
