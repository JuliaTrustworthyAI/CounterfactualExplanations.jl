import DifferentiationInterface as DI

"""
    grad_loss(
        generator::AbstractGradientBasedGenerator,
        ce::AbstractCounterfactualExplanation
    )

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators.
"""
function grad_loss(
    generator::AbstractGradientBasedGenerator,
    ce::AbstractCounterfactualExplanation;
    backend=get_global_ad_backend(),
)

    # Get linear predictions:
    ce_state = ce.counterfactual_state

    # Get target outcome:
    y = ce.target_encoded

    # Create closure
    function loss_wrt_state(x)
        generator.loss(logits(ce.M, CounterfactualExplanations.decode_state(ce, x)), y)
    end

    # Compute gradient:
    g = DI.gradient(loss_wrt_state, backend, ce_state)

    return g
end

"""
    grad_pen(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access. 

If the penalty is not provided, it returns 0.0. By default, Zygote never works out the gradient for constants and instead returns 'nothing', so we need to add a manual step to override this behaviour. See here: https://discourse.julialang.org/t/zygote-gradient/26715.
"""
function grad_pen(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    if isnothing(generator.penalty)
        return 0.0
    else
        _grad = Flux.gradient(ce -> h(generator, ce), ce)[1][:counterfactual_state]
        return _grad
    end
end

"""
    grad_search_opt(
        generator::AbstractGradientBasedGenerator,
        ce::AbstractCounterfactualExplanation,
    )

The default method to compute the gradient of the counterfactual search objective for gradient-based generators.
It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
If the counterfactual is being generated using Probe, the hinge loss is added to the gradient.
"""
function grad_search_opt(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    _grad_loss = grad_loss(generator, ce)
    # println("Loss:")
    # display(grad_loss)
    _grad_pen = grad_pen(generator, ce)
    # println("Penality:")
    # display(grad_pen)
    return _grad_loss .+ _grad_pen
end
