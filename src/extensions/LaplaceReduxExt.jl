"""
    LaplaceReduxModel

Concrete type for neural networks with Laplace Approximation from the `LaplaceRedux` package. Currently subtyping the `AbstractFluxNN` model type, although this may be changed to MLJ in the future.
"""
struct LaplaceReduxModel <: Models.AbstractFluxNN end

Models.all_models_catalogue[:LaplaceReduxModel] = CounterfactualExplanations.LaplaceReduxModel
