``` @meta
CurrentModule = CounterfactualExplanations 
```

# MNIST

In this example we will see how different counterfactual generators can be used to explain deep learning models for image classification. In particular, we will look at MNIST data and visually inspect how the different generators perturb images of handwritten digits in order to change the predicted label to a target label. [Figure 1](#fig-samples) shows a random sample of handwritten digits.

![Figure 1: A few random handwritten digits.](www/mnist_samples.png)

## Pre-trained classifiers

Next we will load two pre-trained deep-learning classifiers:

1.  Simple MLP - `model`
2.  Deep ensemble - `ensemble`

``` julia
using Flux
using CounterfactualExplanations.Data: mnist_data, mnist_model, mnist_ensemble
X, ys = mnist_data()
model = mnist_model()
ensemble = mnist_ensemble()
```

The following code just prepares the models to be used with `CounterfactualExplanations.jl`:

``` julia
using CounterfactualExplanations, CounterfactualExplanations.Models
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

# MLP:
# Step 1)
struct NeuralNetwork <: Models.AbstractFittedModel
    nn::Any
end
# Step 2)
logits(M::NeuralNetwork, X::AbstractArray) = M.nn(X)
probs(M::NeuralNetwork, X::AbstractArray)= softmax(logits(M, X))
M = NeuralNetwork(model)

# Deep ensemble:
# Step 1)
struct FittedEnsemble <: Models.AbstractFittedModel
    ensemble::AbstractArray
end
# Step 2)
using Statistics
logits(M::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([nn(X) for nn in M.ensemble],3), dims=3)
probs(M::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([softmax(nn(X)) for nn in M.ensemble],3),dims=3)
M_ensemble=FittedEnsemble(ensemble)
```

## Generating counterfactuals

We will look at four different approaches here:

1.  Generic approach for the MLP (Wachter, Mittelstadt, and Russell 2017).
2.  Greedy approach for the MLP.
3.  Generic approach for the deep ensemble.
4.  Greedy approach for the deep ensemble (Schut et al. 2021).

They can be implemented using the `GenericGenerator` and the `GreedyGenerator`.

### Turning a 9 into a 4

We will start with an example that should yield intuitive results: the process of turning a handwritten 9 in [Figure 2](#fig-nine) into a 4 is straight-forward for a human - just erase the top part. Let’s see how the different algorithmic approaches perform. First, we preprocess the data below, where we impose that the features (pixel values) are constrained to the follwoing domain: 𝒳 = \[0,1\] ⊂ ℝ.

``` julia
counterfactual_data = CounterfactualData(X,ys';domain=(0,1))
```

Next we choose a random sample for which we will generate counterfactuals in the following:

``` julia
# Randomly selected factual:
using Random
Random.seed!(1234)
x = Flux.unsqueeze(select_factual(counterfactual_data, rand(1:size(X)[2])),2)
target = 5
γ = 0.95
```

![Figure 2: A random handwritten 9.](www/mnist_original.png)

The code below implements the four different approaches one by one. [Figure 3](#fig-example) shows the resulting counterfactuals. In every case the desired label switch is achieved, that is the corresponding classifier classifies the counterfactual as a four. But arguably from a human perspective only the counterfactuals for the deep ensemble look like a 4. For the MLP, both the generic and the greedy approach generate counterfactuals that look much like adversarial examples.

``` julia
# Generic - MLP
generator = GenericGenerator(;loss=:logitcrossentropy)
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ)
img = convert2image(reshape(counterfactual.x′,Int(√(input_dim)),Int(√(input_dim))))
plt_wachter = plot(img, title="MLP - Wachter")

# Generic - Deep Ensemble
counterfactual = generate_counterfactual(x, target, counterfactual_data, M_ensemble, generator; γ=γ)
img = convert2image(reshape(counterfactual.x′,Int(√(input_dim)),Int(√(input_dim))))
plt_wachter_de = plot(img, title="Ensemble - Wachter")

# Greedy - MLP
generator = GreedyGenerator(;loss=:logitcrossentropy)
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator; γ=γ)
img = convert2image(reshape(counterfactual.x′,Int(√(input_dim)),Int(√(input_dim))))
plt_greedy = plot(img, title="MLP - Greedy")

# Greedy - Deep Ensemble
counterfactual = generate_counterfactual(x, target, counterfactual_data, M_ensemble, generator; γ=γ)
img = convert2image(reshape(counterfactual.x′,Int(√(input_dim)),Int(√(input_dim))))
plt_greedy_de = plot(img, title="Ensemble - Greedy")

plt_list = [plt_orig, plt_wachter, plt_greedy, plt_wachter_de, plt_greedy_de]
plt = plot(plt_list...,layout=(1,length(plt_list)),axis=nothing, size=(1200,240))
savefig(plt, joinpath(www_path, "MNIST_9to4.png"))
```

![Figure 3: Counterfactual explanations for MNIST data: turning a 9 into a 4](www/MNIST_9to4.png)

### References

Schut, Lisa, Oscar Key, Rory Mc Grath, Luca Costabello, Bogdan Sacaleanu, Yarin Gal, et al. 2021. “Generating Interpretable Counterfactual Explanations by Implicit Minimisation of Epistemic and Aleatoric Uncertainties.” In *International Conference on Artificial Intelligence and Statistics*, 1756–64. PMLR.

Wachter, Sandra, Brent Mittelstadt, and Chris Russell. 2017. “Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR.” *Harv. JL & Tech.* 31: 841.
