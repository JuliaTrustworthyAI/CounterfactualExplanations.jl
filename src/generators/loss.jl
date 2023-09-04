"""
    ℓ(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)

Dispatches to the appropriate loss function for any generator.
"""
function ℓ(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)
    return ℓ(generator, generator.loss, ce)
end

"""
    ℓ(generator::AbstractGenerator, loss::Nothing, ce::AbstractCounterfactualExplanation)

Overloads the `ℓ` function for the case where no loss function is provided.
"""
function ℓ(
    generator::AbstractGenerator, loss::Nothing, ce::AbstractCounterfactualExplanation
)
    return CounterfactualExplanations.guess_loss(ce)(ce)
end

"""
    ℓ(generator::AbstractGenerator, loss::Function, ce::AbstractCounterfactualExplanation)

Overloads the `ℓ` function for the case where a single loss function is provided.
"""
function ℓ(
    generator::AbstractGenerator, loss::Function, ce::AbstractCounterfactualExplanation
)
    return loss(ce)
end
