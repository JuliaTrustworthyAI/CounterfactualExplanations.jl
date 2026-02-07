import DifferentiationInterface as DI

"""a require"cmp.utils.feedkeys".run(265)

    grad_loss(
        generator::AbstractGradientBasedGenerator,
        ce::AbstractCounterfactualExplanation
    )

The da require"cmp.utils.feedkeys".run(262)
efault method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators.
"""
function grad_loss(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation;
)

    # Get counterfactual state: 
    ce_state = ce.counterfactual_state

    # Get target outcome:
    y = ce.target_encoded

    # Get AD backend:
    backend = CounterfactualExplanations.choose_ad_backend(ce)

    # Create closure
    function loss_wrt_state(x)
        return generator.loss(
            logits(ce.M, CounterfactualExplanations.decode_state(ce, x)), y
        )
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
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation;
)
    if isnothing(generator.penalty)
        return 0.0
    else
        # Get counterfactual state: 
        ce_state = ce.counterfactual_state
        pen = ifelse(generator.penalty isa Function, [generator.penalty], generator.penalty)

        # Get AD backend:
        backend = CounterfactualExplanations.choose_ad_backend(ce)

        # Create closure
        function pen_wrt_state(x)
            cf = CounterfactualExplanations.decode_state(ce, x)
            if pen isa Vector{<:Tuple}
                # Keyword arguments supplied:
                sum(generator.λ .* [fun(cf, ce; kwargs...) for (fun, kwargs) in pen])
            else
                # No keyword arguments supplied:
                sum(generator.λ .* [fun(cf, ce) for fun in pen])
            end
        end

        # Compute gradient:
        g = DI.gradient(pen_wrt_state, backend, ce_state)

        return g
    end
end

"""
    grad_search_opt(
        generator::AbstractGradientBasedGenerator,
        ce::AbstractCounterfactualExplanation,
    )

The default method to compute the gradient of the counterfactual search objective for gradient-based generators.
It simply computes the weighted sum over partial derivatives. It assumes that `Zygote.jl` has gradient access.
If the counterfactual is being generated using Probe, the hinge loss is added to the gradient.
"""
function grad_search_opt(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    _grad_loss = grad_loss(generator, ce)
    _grad_pen = grad_pen(generator, ce)
    return _grad_loss .+ _grad_pen
end
