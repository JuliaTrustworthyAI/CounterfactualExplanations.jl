"""
    LaplaceNN

Concrete type for neural networks with Laplace Approximation from the `LaplaceRedux` package. Currently subtyping the `FluxNN` model type, although this may be changed to MLJ in the future.
"""
struct LaplaceNN <: Models.FluxNN end

Models.all_models_catalogue[:LaplaceNN] = CounterfactualExplanations.LaplaceNN
