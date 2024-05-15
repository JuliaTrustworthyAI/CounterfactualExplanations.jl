"""
    EvoTreeModel

Exposes the `EvoTreeModel` from the `EvoTreesExt` extension.
"""
struct EvoTree <: Models.MLJModelType end

CounterfactualExplanations.Models.all_models_catalogue[:EvoTree] = EvoTree
