# Loss:
"""
    ℓ(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(
    generator::AbstractGenerator,
    ce::AbstractCounterfactualExplanation,
)
    loss_fun = if !isnothing(generator.loss)
        generator.loss
    else
        CounterfactualExplanations.guess_loss(ce)
    end
    @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
    loss = loss_fun(ce)
    return loss
end

# Complexity:
"""
    h(generator::AbstractGenerator, ce::AbstractCounterfactualExplanation)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(
    generator::AbstractGenerator,
    ce::AbstractCounterfactualExplanation,
)
    if isnothing(generator.penalty)
        penalty = 0.0
    elseif typeof(generator.penalty) <: Vector
        cost = [fun(ce) for fun in generator.penalty]
    else
        cost = generator.penalty(ce)
    end
    penalty = sum(generator.λ .* cost)
    return penalty
end
