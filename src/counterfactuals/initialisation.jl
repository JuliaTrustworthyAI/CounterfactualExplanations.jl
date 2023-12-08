"""
    initialize_state(ce::CounterfactualExplanation)

Initializes the starting point for the factual(s):
    
1. If `ce.initialization` is set to `:identity` or counterfactuals are searched in a latent space, then nothing is done.
2. If `ce.initialization` is set to `:add_perturbation`, then a random perturbation is added to the factual following following Slack (2021): https://arxiv.org/abs/2106.02666. The authors show that this improves adversarial robustness.
"""
function initialize_state(ce::CounterfactualExplanation)
    @assert ce.initialization ∈ [:identity, :add_perturbation]
    s′ = ce.s′

    # Add random perturbation following Slack (2021): https://arxiv.org/abs/2106.02666
    if ce.initialization == :add_perturbation
        Δs′ = randn(eltype(s′), size(s′)) * convert(eltype(s′), 0.1)
        Δs′ = apply_mutability(ce, Δs′)
        s′ .+= Δs′
    end

    return s′
end
