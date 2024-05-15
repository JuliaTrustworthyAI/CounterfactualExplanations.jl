using Tables

"Abstract type for MLJ models."
abstract type MLJModelType <: AbstractModelType end

"""
    train(
        M::Model,
        type::MLJModelType,
        data::CounterfactualData,
    )

Overloads the `train` function for MLJ models.
"""
function train(
    M::Model,
    type::MLJModelType,
    data::CounterfactualData,
)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    if M.likelihood ∉ [:classification_multi, :classification_binary]
        y = float.(y.refs)
    end
    X = MLJBase.reformat(M.model, X)[1]
    X = Tables.table(X)
    mach = MLJBase.machine(M.model, X, y)
    MLJBase.fit!(mach)
    M.fitresult = mach.fitresult
    return M
end

"""
    probs(
        M::Model,
        type::MLJModelType,
        X::AbstractArray,
    )

Overloads the [`probs`](@ref) method for MLJ models. 

## Note for developers

Note that currently the underlying MLJ methods (`reformat`, `predict`) are incompatible with Zygote's autodiff. For differentiable MLJ models, the [`probs``](@ref) and [`logits`](@ref) methods need to be overloaded.
"""
function probs(
    M::Model,
    type::MLJModelType,
    X::AbstractArray,
)
    if ndims(X)==1
        X = X[:,:]      # account for 1-dimensional inputs
    end
    X = Tables.table(X)
    X = MLJBase.reformat(M.model, X)[1]
    output = MLJBase.predict(M.model, M.fitresult, X)
    p = MLJBase.pdf(output, MLJBase.classes(output))'
    if M.likelihood == :classification_binary
        p = reshape(p[2, :], 1, size(p, 2))
    end
    return p
end

"""
    logits(M::Model, type::MLJModelType, X::AbstractArray)

Overloads the [logits](@ref) method for MLJ models.
"""
function logits(M::Model, type::MLJModelType, X::AbstractArray)
    p = probs(M, type, X)
    if M.likelihood == :classification_binary
        output = log.(p ./ (1 .- p))
    else
        output = log.(p)
    end
    return output
end
