using CounterfactualExplanations.Generators

"""
    Generators.∂ℓ(
        generator::AbstractGradientBasedGenerator,
        model::RTorchModel,
        ce::AbstractCounterfactualExplanation,
    )

Extends the `∂ℓ` function for gradient-based generators operating on RTorch models.
"""
function Generators.∂ℓ(
    generator::AbstractGradientBasedGenerator,
    model::RTorchModel,
    ce::AbstractCounterfactualExplanation,
)
    x = ce.x
    target = Float32.(ce.target_encoded)

    model_nn = model.nn

    R"""
    x <- torch_tensor(t($x), requires_grad=TRUE)
    target <- torch_tensor(t($target))
    output <- $model_nn(x)
    obj_loss <- nnf_binary_cross_entropy_with_logits(output, target)
    obj_loss$backward()
    """

    grad = RCall.rcopy(R"t(as_array(x$grad))")
    grad = Float32.(grad)

    return grad
end
