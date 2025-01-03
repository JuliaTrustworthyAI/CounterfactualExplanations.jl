# Setup
M = synthetic[:classification_binary][:models][:MLP][:model]
counterfactual_data = synthetic[:classification_binary][:data]
X = counterfactual_data.X
ys = counterfactual_data.y
generator = Generators.generator_catalogue[:revise]()

# Counterfactual search
x = DataPreprocessing.select_factual(counterfactual_data, Random.rand(1:size(X, 2)))
y = Models.predict_label(M, counterfactual_data, x)
target = get_target(counterfactual_data, y[1])
ce = CounterfactualExplanations.generate_counterfactual(
    x, target, counterfactual_data, M, generator
)

using CounterfactualExplanations.GenerativeModels: train!, retrain!, VAE
generative_model = VAE(size(X, 1))
train!(generative_model, X)
retrain!(generative_model, X)

@test true
