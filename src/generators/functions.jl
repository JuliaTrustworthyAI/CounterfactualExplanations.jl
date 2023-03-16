# Loss:
"""
    ℓ(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(
    generator::AbstractGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)

    loss_fun =
        !isnothing(generator.loss) ? generator.loss :
        CounterfactualExplanations.guess_loss(counterfactual_explanation)
    @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
    loss = loss_fun(counterfactual_explanation)
    return loss
end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(
    generator::AbstractGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation
)
    if isnothing(generator.penalty)
        penalty = 0.0
    elseif typeof(generator.penalty) <: Vector
        cost = [fun(counterfactual_explanation) for fun in generator.penalty]
    else
        cost = generator.penalty(counterfactual_explanation)
    end
    penalty = sum(generator.λ .* cost)
    return penalty
end
