``` @meta
CurrentModule = CounterfactualExplanations 
```

# Multi-class targets

In the existing literature counterfactual explanations have typically been applied in the binary classification setting (Verma, Dickerson, and Hines 2020). Research on algorithmic recourse in particular typically involves real-world datasets with an obvious target class - e.g. individual receives credit - and an adverse outcome - e.g. individual is denied loan (Karimi et al. 2020). Still, counterfactual explanations are very much also applicable in the multi-class setting. In this tutorial we will go through an illustrative example involving the toy dataset shown in [Figure 1](#fig-multi) below.

``` julia
using CounterfactualExplanations.Data
xs, ys = Data.toy_data_multi()
X = hcat(xs...)
y_train = Flux.onehotbatch(ys, unique(ys))
y_train = Flux.unstack(y_train',1)
```

![Figure 1: Synthetic dataset containing four different classes.](www/multi_samples.png)

## Classifier

To classify the data we use a simple multi-layer perceptron (MLP). In this case the MLP outputs four logits, one for each class. Contrary to the binary setting we therefore choose `logitcrossentropy` as our loss functiona as opposed to `logitbinarycrossentropy`.

``` julia
n_hidden = 32
out_dim = length(unique(ys))
kw = (output_dim=out_dim, dropout=true)
nn = build_model(;kw...)
loss(x, y) = Flux.Losses.logitcrossentropy(nn(x), y)
ps = Flux.params(nn)
data = zip(xs,y_train)
```

The following code just trains the neural network for the task:

``` julia
using Flux.Optimise: update!, ADAM
opt = ADAM()
epochs = 10
avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))
show_every = epochs/10

for epoch = 1:epochs
  for d in data
    gs = gradient(params(nn)) do
      l = loss(d...)
    end
    update!(opt, params(nn), gs)
  end
  if epoch % show_every == 0
    println("Epoch " * string(epoch))
    @show avg_loss(data)
  end
end
```

To make the model compatible with our package we need to 1) declare it as a subtype of `Models.AbstractFittedModel` and 2) dispatch the relevant methods. Logits are returned by the model on call and passed through the softmax function to generate the vector of class probabilities.

``` julia
using CounterfactualExplanations, CounterfactualExplanations.Models
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

# Step 1)
struct NeuralNetwork <: Models.AbstractFittedModel
    model::Any
end

# Step 2)
logits(M::NeuralNetwork, X::AbstractArray) = M.model(X)
probs(M::NeuralNetwork, X::AbstractArray)= softmax(logits(M, X))
M = NeuralNetwork(nn);
```

[Figure 2](#fig-multi-contour) shows the resulting class probabilities in the two-dimensional feature domain.

![Figure 2: Class probabilities for MLP.](www/multi_contour.png)

# Generating counterfactuals

We first preprocess the data:

``` julia
counterfactual_data = CounterfactualData(X,ys')
```

Next we randomly select an individual sample from any class and choose any of the remaining classes as our target at random.

``` julia
# Randomly selected factual:
Random.seed!(42)
x = select_factual(counterfactual_data, rand(1:size(X)[2])) 
y = Flux.onecold(probs(M, x),unique(ys))
target = rand(unique(ys)[1:end .!= y]) # opposite label as target
```

Generic counterfactual search can then be implemented as follows. The only difference to the binary setting is that we need to declare `logitcrossentropy` as the loss function for the counterfactual search. [Figure 3](#fig-multi-generic) shows the resulting counterfactual path.

``` julia
# Define generator:
generator = GenericGenerator(;loss=:logitcrossentropy)
# Generate recourse:
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
```

![Figure 3: Counterfactual path for generic generator.](www/multi_generic_recourse.gif)

## Deep ensemble

Staying consistent with previous tutorial we will also briefly look at the Bayesian setting. To incorporate uncertainty we use a simple deep ensemble instead of a single MLP.

``` julia
ensemble = build_ensemble(5;kw=(output_dim=out_dim,))
ensemble, = forward(ensemble, data, opt, n_epochs=epochs, plot_loss=false)
```

As before, we need to subtype and disptach:

``` julia
# Step 1)
struct FittedEnsemble <: Models.AbstractFittedModel
    ensemble::AbstractArray
end

# Step 2)
using Statistics
logits(M::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([nn(X) for nn in M.ensemble],3), dims=3)
probs(M::FittedEnsemble, X::AbstractArray) = mean(Flux.stack([softmax(nn(X)) for nn in M.ensemble],3),dims=3)

M=FittedEnsemble(ensemble)
```

[Figure 4](#fig-multi-ensemble-contour) shows the resulting class probabilities.

![Figure 4: Class probabilities for deep ensemble.](www/multi_ensemble_contour.png)

For the greedy recourse generator we also specify `logitcrossentropy` as our loss function and modify the hyperparameters slightly. [Figure 5](#fig-greedy) shows the resulting counterfactual path.

``` julia
generator = GreedyGenerator(loss=:logitcrossentropy,δ=0.25,n=20)
counterfactual = generate_counterfactual(x, target, counterfactual_data, M, generator)
```

![Figure 5: Counterfactual path for greedy generator.](www/multi_greedy_recourse.gif)

# References

Karimi, Amir-Hossein, Gilles Barthe, Bernhard Schölkopf, and Isabel Valera. 2020. “A Survey of Algorithmic Recourse: Definitions, Formulations, Solutions, and Prospects.” *arXiv Preprint arXiv:2010.04050*.

Verma, Sahil, John Dickerson, and Keegan Hines. 2020. “Counterfactual Explanations for Machine Learning: A Review.” *arXiv Preprint arXiv:2010.10596*.
