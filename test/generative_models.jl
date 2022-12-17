using MLUtils

# Setup
M = LogisticModel([1 1], [0])
generator = generator_catalog[:revise]()
X, ys = toy_data_linear()
X = MLUtils.stack(X, dims = 2)
counterfactual_data = CounterfactualData(X, ys')
# Coutnerfactual search
x = select_factual(counterfactual_data, rand(1:size(X, 2)))
p_ = probs(M, x)
y = round(p_[1])
target = y == 0 ? 1 : 0
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)

using CounterfactualExplanations.GenerativeModels: retrain!
CounterfactualExplanations.GenerativeModels.retrain!(
    counterfactual_data.generative_model,
    X,
    ys,
)
