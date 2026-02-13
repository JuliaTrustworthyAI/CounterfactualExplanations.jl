import DifferentiationInterface as DI

function loss_wrt_state(x::AbstractArray, ce::AbstractCounterfactualExplanation)
    return ce.generator.loss(
        logits(ce.M, CounterfactualExplanations.decode_state(ce, x)), ce.target_encoded
    )
end

"""
    grad_loss(
        generator::AbstractGradientBasedGenerator,
        ce::AbstractCounterfactualExplanation
    )

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators.
"""
function grad_loss(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation;
)

    # Get counterfactual state: 
    ce_state = ce.counterfactual_state

    # Get AD backend:
    backend = get!(ce.search, :ad_backend, CounterfactualExplanations.choose_ad_backend(ce))

    # Get prepared gradient:
    prep = get!(
        ce.search,
        :prep_loss,
        DI.prepare_gradient(loss_wrt_state, backend, similar(ce_state), DI.Constant(ce)),
    )

    # Compute gradient:
    g = DI.gradient(loss_wrt_state, prep, backend, ce_state, DI.Constant(ce))

    return g
end

function pen_wrt_state(x::AbstractArray, ce::AbstractCounterfactualExplanation)
    return pen_wrt_state(x, ce.generator.penalty, ce)
end

function pen_wrt_state(
    x::AbstractArray, pen::Nothing, ce::AbstractCounterfactualExplanation
)
    return 0.0
end

function pen_wrt_state(
    x::AbstractArray, pen::PenaltyOrFun, ce::AbstractCounterfactualExplanation
)
    return ce.generator.λ * pen(x, ce)
end

function pen_wrt_state(
    x::AbstractArray, pen::Vector{<:Tuple}, ce::AbstractCounterfactualExplanation
)
    total = zero(eltype(ce.generator.λ))
    @inbounds for i in 1:length(pen)
        fun, kwargs = pen[i]
        total += ce.generator.λ[i] * fun(x, ce; kwargs...)
    end
    return total
end

function pen_wrt_state(
    x::AbstractArray, pen::Vector{<:PenaltyOrFun}, ce::AbstractCounterfactualExplanation
)
    total = zero(eltype(ce.generator.λ))
    @inbounds for i in eachindex(pen, ce.generator.λ)
        total += ce.generator.λ[i] * pen[i](x, ce)
    end
    return total
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

    # Get counterfactual state: 
    ce_state = ce.counterfactual_state

    # Get AD backend:
    backend = get!(ce.search, :ad_backend, CounterfactualExplanations.choose_ad_backend(ce))

    # Get prepared gradient:
    prep = get!(
        ce.search,
        :prep_pen,
        DI.prepare_gradient(pen_wrt_state, backend, similar(ce_state), DI.Constant(ce)),
    )

    # Compute gradient:
    g = DI.gradient(pen_wrt_state, prep, backend, ce_state, DI.Constant(ce))

    return g
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
