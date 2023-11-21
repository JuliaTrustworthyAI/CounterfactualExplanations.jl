"""
    EvoTreeModel

Exposes the `EvoTreeModel` from the `EvoTreesExt` extension.
"""
function EvoTreeModel end
export EvoTreeModel

CounterfactualExplanations.Models.all_models_catalogue[:EvoTree] = EvoTreeModel
CounterfactualExplanations.Models.mlj_models_catalogue[:EvoTree] = EvoTreeModel
