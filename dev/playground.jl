using CounterfactualExplanations
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DecisionTree
using Plots
using TaijaData
using TaijaPlotting

# Counteractual data and model:
n = 3000
data = CounterfactualData(load_moons(n; noise=0.3)...)
M = fit_model(data, :MLP)
target = 0
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, data, M, generator)

# Surrogate:
generator = Generators.TCRExGenerator(0.02)
tree, fitresult = Generators.grow_surrogate(generator, ce)
M = CounterfactualExplanations.DecisionTreeModel(tree; fitresult=fitresult)
plot(M, data; ms=3, markerstrokewidth=0, size=(500,500), colorbar=false)