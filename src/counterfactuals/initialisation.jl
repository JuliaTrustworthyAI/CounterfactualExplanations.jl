using Flux: Flux

"""
    initialize_state(ce::CounterfactualExplanation)

Initializes the starting point for the factual(s):
    
1. If `ce.initialization` is set to `:identity` or counterfactuals are searched in a latent space, then nothing is done.
2. If `ce.initialization` is set to `:add_perturbation`, then a random perturbation is added to the factual following following Slack (2021): https://arxiv.org/abs/2106.02666. The authors show that this improves adversarial robustness.
"""
function initialize_state(ce::CounterfactualExplanation)
    @assert ce.initialization ∈ [:identity, :add_perturbation]

    counterfactual_state = ce.counterfactual_state

    # No perturbation:
    if ce.initialization == :identity
        return counterfactual_state
    end

    # If latent space, initial point is random anyway:
    if ce.generator.latent_space
        return counterfactual_state
    end

    # Add random perturbation following Slack (2021): https://arxiv.org/abs/2106.02666
    if ce.initialization == :add_perturbation
        Δcounterfactual_state =
            randn(eltype(counterfactual_state), size(counterfactual_state)) *
            convert(eltype(counterfactual_state), 0.1)
        Δcounterfactual_state = apply_mutability(ce, Δcounterfactual_state)
        counterfactual_state .+= Δcounterfactual_state
    end

    return counterfactual_state
end

"""
    initialize_state!(ce::CounterfactualExplanation)

Initializes the starting point for the factual(s) in-place.
"""
function initialize_state!(ce::CounterfactualExplanation)
    ce.counterfactual_state = initialize_state(ce)

    return ce
end
