"""
    LaplaceReduxModel

Concrete type for neural networks with Laplace Approximation from the `LaplaceRedux` package. Currently subtyping the `AbstractFluxNN` model type, although this may be changed to MLJ in the future.
"""
struct LaplaceReduxModel <: Models.AbstractFluxNN end

Models.all_models_catalogue[:LaplaceReduxModel] =
    CounterfactualExplanations.LaplaceReduxModel

"The `LaplaceReduxModel` model type requires `ForwardDiff`"
function Objectives.ADRequirements(::Type{<:LaplaceReduxModel})
    if VERSION >= v"1.12"
        @warn "Zygote support for `LaplaceRedux` model is broken on Julia v$VERSION. Falling back to ForwardDiff, which can lead to slower performance for high-dimensional inputs." maxlog =
            1
        Objectives.NeedsForwardDiff()
    else
        Objectives.NoADRequirements()
    end
end
