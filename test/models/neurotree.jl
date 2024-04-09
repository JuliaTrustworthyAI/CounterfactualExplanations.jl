using CounterfactualExplanations
using CounterfactualExplanations.Models
using MLJBase
using NeuroTreeModels
using TaijaData

m = NeuroTreeRegressor(; depth=5, nrounds=10)
data = CounterfactualData(load_overlapping()...)
M = fit_model(data, :NeuroTree; outsize=2, depth=4, lr=2e-2)
predict_label(M, data)
# mach = machine(m, X, y) 
# MLJBase.fit!(mach)
