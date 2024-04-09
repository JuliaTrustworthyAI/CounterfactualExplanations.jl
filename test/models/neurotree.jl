using CounterfactualExplanations
using CounterfactualExplanations.Models
using MLJBase
using NeuroTreeModels
using TaijaData

m = NeuroTreeRegressor(; depth=5, nrounds=10)
data = CounterfactualData(load_overlapping()...)
M = fit_model(data, :NeuroTree)
# mach = machine(m, X, y) 
# MLJBase.fit!(mach)