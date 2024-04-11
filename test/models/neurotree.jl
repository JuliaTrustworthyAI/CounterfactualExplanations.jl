using CounterfactualExplanations
using CounterfactualExplanations.Models
using MLJBase
using NeuroTreeModels
using Plots
using TaijaData
using TaijaPlotting

m = NeuroTreeRegressor(; depth=5, nrounds=10)
data = CounterfactualData(load_linearly_separable()...)
M = fit_model(data, :NeuroTree; outsize=2, depth=4, lr=2e-2, nrounds=100)

target = 2
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

generator = GenericGenerator(; opt=Descent(0.1))
ce = generate_counterfactual(x, target, data, M, generator)
plot(ce; alpha=0.1)