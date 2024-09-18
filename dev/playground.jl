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

# Generic search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, data, M, generator)

# T-CREx ###################################################################
ρ = 0.02
τ = 0.9
generator = Generators.TCRExGenerator(ρ)

# (a) ##############################

# Surrogate:
model, fitresult = Generators.grow_surrogate(generator, ce)
M = CounterfactualExplanations.DecisionTreeModel(model; fitresult=fitresult)
plot(M, data; ms=3, markerstrokewidth=0, size=(500,500), colorbar=false)

# Extract rules:
x = Generators.extract_rules(fitresult[1])
print_tree(fitresult[1])

# Compute feasibility and accuracy:
feas = Generators.rule_feasibility.(x, (ce.data.X,))
@assert minimum(feas) >= ρ
acc_0 = Generators.rule_accuracy.(x, (X,), (fx,), (0,))
acc_1 = Generators.rule_accuracy.(x, (X,), (fx,), (1,))
@assert all(acc_0 .+ acc_1 .== 1.0)

# (b) ##############################