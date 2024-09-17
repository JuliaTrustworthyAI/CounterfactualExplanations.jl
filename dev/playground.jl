using CounterfactualExplanations
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DecisionTree
using TaijaData

# Counteractual data and model:
n = 1000
data = CounterfactualData(load_moons(n)...)
M = fit_model(data, :MLP)
target = 0
factual = 1
chosen = rand(findall(predict_label(M, data) .== factual))
x = select_factual(data, chosen)

# Search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, data, M, generator)