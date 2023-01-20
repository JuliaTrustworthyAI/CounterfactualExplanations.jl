using MLUtils

# Setup
M = synthetic[:classification_binary][:models][:MLP][:model]
counterfactual_data = synthetic[:classification_binary][:data]
X = counterfactual_data.X
ys = counterfactual_data.y
generator = generator_catalog[:revise]()

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
