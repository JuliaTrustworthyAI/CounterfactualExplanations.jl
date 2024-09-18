using CounterfactualExplanations
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DecisionTree
using Plots
using TaijaData
using TaijaPlotting

# Counteractual data and model:
n = 3000
data = CounterfactualData(load_moons(n; noise=0.4)...)
M = fit_model(data, :MLP)
target = 0
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, data, M, generator)

# Surrogate:
ρ = 0.02
generator = Generators.TCRExGenerator(ρ)
model, fitresult = Generators.grow_surrogate(generator, ce)
M = CounterfactualExplanations.DecisionTreeModel(model; fitresult=fitresult)
plot(M, data; ms=3, markerstrokewidth=0, size=(500,500), colorbar=false)

# Extract rules:
x = Generators.extract_rules(fitresult[1])
print_tree(fitresult[1])
feas = Generators.rule_feasibility.(x, (ce.data.X,))
@assert minimum(feas) >= ρ