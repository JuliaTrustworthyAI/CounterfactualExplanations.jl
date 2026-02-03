"""
    lossfun(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)

Dispatches to the appropriate loss function for any generator.
"""
function lossfun(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)
    return lossfun(generator, generator.loss, ce)
end

"""
    lossfun(generator::AbstractGenerator, loss::Nothing, ce::AbstractCounterfactualExplanation)

Overloads the `lossfun` function for the case where no loss function is provided.
"""
function lossfun(
    generator::AbstractGenerator, loss::Nothing, ce::AbstractCounterfactualExplanation
)
    return CounterfactualExplanations.guess_loss(ce)(ce)
end

"""
    lossfun(generator::AbstractGenerator, loss::Function, ce::AbstractCounterfactualExplanation)

Overloads the `lossfun` function for the case where a single loss function is provided.
"""
function lossfun(
    generator::AbstractGenerator, loss::Function, ce::AbstractCounterfactualExplanation
)
    return loss(ce)
end
