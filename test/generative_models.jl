using MLUtils

# Setup
M = synthetic[:classification_binary][:models][:flux][:model]
generator = generator_catalog[:revise]()
xs, ys = (synthetic[:classification_binary][:data][:xs], synthetic[:classification_binary][:data][:ys])
X = MLUtils.stack(xs, dims = 2)
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
