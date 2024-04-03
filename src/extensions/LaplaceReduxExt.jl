"""
    LaplaceReduxModel

Exposes the `LaplaceReduxModel` from the `LaplaceReduxExt` extension.
"""
function LaplaceReduxModel end
export LaplaceReduxModel

CounterfactualExplanations.Models.all_models_catalogue[:LaplaceRedux] = LaplaceReduxModel
CounterfactualExplanations.Models.standard_models_catalogue[:LaplaceRedux] =
    LaplaceReduxModel
