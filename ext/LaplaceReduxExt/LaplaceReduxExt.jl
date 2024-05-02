module LaplaceReduxExt

using CounterfactualExplanations
using CounterfactualExplanations.Models: Models
using LaplaceRedux: LaplaceRedux

include("laplace_redux.jl")

function Models.fit_model(data::CounterfactualData, model::typeof(LaplaceReduxModel))
    return Models.fit_model(data, model)
end

end
