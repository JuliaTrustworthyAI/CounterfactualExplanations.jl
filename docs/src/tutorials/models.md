```@meta
CurrentModule = CounterfactualExplanations 
```

# Models

## Default models

There are currently structures for two default models that can be used with CounterfactualExplanations.jl:

1. [`LogisticModel(w::AbstractArray,b::AbstractArray)`](@ref)
2. [`BayesianLogisticModel(μ::AbstractArray,Σ::AbstractArray)`](@ref)

Both take sets of estimated parameters at the point of instantiation: the constructors will not fit a model for you, but assume that you have already estimated the respective model yourself and have access to its parameter estimates. Based on the supplied parameters methods to predict logits and probabilities are already implemented and used in the counterfactual search. 

For the simple logistic regression model logits are computed as $a=Xw + b$ and probabilities are simply $\sigma(a)$. For the Bayesian logistic regression model logits are computed as $X\mu$ and the predictive posterior is computed through Laplace approximation.

## Custom models

Apart from the default models you can use any arbitrary (differentiable) model and generate recourse in the same way as before. Only two steps are necessary to make your own model compatible with CounterfactualExplanations.jl:

1. The model needs to be declared as a subtype of `CounterfactualExplanations.Models.AbstractFittedModel`.
2. You need to extend the functions `CounterfactualExplanations.Models.logits` and `CounterfactualExplanations.Models.probs` to accept your custom model.

Below we will go through a simple example to see how this can be done in practice. 

### Neural network

In this example we will build a simple artificial neural network using [Flux.jl](https://fluxml.ai/) for a binary classification task.


```julia
# Import libraries.
using Flux, Plots, Random, PlotThemes, Statistics, CounterfactualExplanations
theme(:wong)
using Logging
disable_logging(Logging.Info)
```


    LogLevel(1)


First we generate some toy data below. The code that generates this data was borrowed from a great tutorial about Bayesian neural networks provided by [Turing.jl](https://turing.ml/dev/), which you may find [here](https://turing.ml/dev/tutorials/03-bayesian-neural-network/). 

The plot below shows the generated samples in the 2D feature space where colours indicate the associated labels. CounterfactualExplanationsly this data is not linearly separable and the default `LogisticModel` would be ill suited for this classification task.


```julia
# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

x, y = toy_data_non_linear(N)
X = hcat(x...)
plt = plot()
plt = plot_data!(plt,X',y);
savefig(plt, "www/models_samples.png")
```

![](www/models_samples.png)

#### Training the model

Instead we will build a simple artificial neural network `nn` with one hidden layer. For additional resources on how to do deep learning with [Flux.jl](https://fluxml.ai/) just have a look at their documentation. 


```julia
nn = build_model(dropout=true,activation=Flux.σ)
loss(x, y) = Flux.Losses.logitbinarycrossentropy(nn(x), y)
ps = Flux.params(nn)
data = zip(x,y);
```

The code below trains the neural network for the task at hand. The plot shows the (training) loss over time. Note that normally we would be interested in loss with respect to a validation data set. But since we are primarily interested in generated recourse for a trained classifier, here we will just keep things very simple.


```julia
using Flux.Optimise: update!, ADAM
opt = ADAM()
epochs = 100
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

    Epoch 10
    avg_loss(data) = 0.6892187490309944
    Epoch 20
    avg_loss(data) = 0.6767634057872471
    Epoch 30
    avg_loss(data) = 0.6557134557550864
    Epoch 40
    avg_loss(data) = 0.6238793502186495
    Epoch 50
    avg_loss(data) = 0.574156178094375
    Epoch 60
    avg_loss(data) = 0.5067736276756599
    Epoch 70
    avg_loss(data) = 0.42957685824823677
    Epoch 80
    avg_loss(data) = 0.3534671179124217
    Epoch 90
    avg_loss(data) = 0.2879001871677585
    Epoch 100
    avg_loss(data) = 0.23253485574062113


#### Generating recourse

Now it's game time: we have a fitted model $M: \mathcal{X} \mapsto y$ and are interested in generating recourse for some individual $\overline{x}\in\mathcal{X}$. As mentioned above we need to do a bit more work to prepare the model to be used by CounterfactualExplanations.jl. 

The code below takes care of all of that: in step 1) it declares our model as a subtype of `Models.AbstractFittedModel` and in step 2) it just extends the two functions. 


```julia
using CounterfactualExplanations, CounterfactualExplanations.Models
import CounterfactualExplanations.Models: logits, probs # import functions in order to extend

# Step 1)
struct NeuralNetwork <: Models.AbstractFittedModel
    nn::Any
end

# Step 2)
logits(𝑴::NeuralNetwork, X::AbstractArray) = 𝑴.nn(X)
probs(𝑴::NeuralNetwork, X::AbstractArray)= σ.(logits(𝑴, X))
𝑴 = NeuralNetwork(nn)
```


    NeuralNetwork(Chain(Dense(2, 32, σ), Dropout(0.1), Dense(32, 1)))


The plot below shows the predicted probabilities in the feature domain. Evidently our simple neural network is doing very well on the training data, as expected. 


```julia
# Plot the posterior distribution with a contour plot.
plt = plot_contour(X',y,𝑴);
savefig(plt, "www/models_contour.png")
```

![](www/models_contour.png)

Now we just select a random sample from our data and based on its current label we set as our target the opposite label and desired threshold for the predicted probability.


```julia
using Random
Random.seed!(123)
x̅ = X[:,rand(1:size(X)[2])]
y̅ = round(probs(𝑴, x̅)[1])
target = ifelse(y̅==1.0,0.0,1.0) # opposite label as target
γ = 0.75; # desired level of confidence
```

Then finally we use the `GenericGenerator` to generate recourse. The plot further below shows the resulting counterfactual path.


```julia
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```


```julia
T = size(recourse.path)[1]
X_path = reduce(hcat,recourse.path)
ŷ = CounterfactualExplanations.target_probs(probs(recourse.𝑴, X_path),target)
p1 = plot_contour(X',y,𝑴;clegend=false, title="MLP")
anim = @animate for t in 1:T
    scatter!(p1, [recourse.path[t][1]], [recourse.path[t][2]], ms=5, color=Int(y̅), label="")
    p2 = plot(1:t, ŷ[1:t], xlim=(0,T), ylim=(0, 1), label="p(y̲=" * string(target) * ")", title="Validity", lc=:black)
    Plots.abline!(p2,0,γ,label="threshold γ", ls=:dash) # decision boundary
    plot(p1,p2,size=(800,400))
end
gif(anim, "www/models_generic_recourse.gif", fps=5);
```

    HI


![](www/models_generic_recourse.gif)

### Ensemble of neural networks

In the context of Bayesian classifiers the `GreedyGenerator` can be used since minimizing the predictive uncertainty acts as a proxy for *realism* and *unambiquity*. In other words, if we have a model that incorporates uncertainty, we can generate realistic counterfactuals without the need for a complexity penalty. 

One efficient way to produce uncertainty estimates in the context of deep learning is to simply use an ensemble of artificial neural networks (also referred to as *deep ensemble*). To this end we can use the `build_model` function from above repeatedly to compose an ensemble of $K$ neural networks:


```julia
𝓜 = build_ensemble(5;kw=(dropout=true,activation=Flux.σ));
```

Now we need to be able to train this ensemble, which boils down to training each neural network separately. For this purpose will just summarize the process for training a single neural network (as per above) in a wrapper function:


```julia
function forward_nn(nn, loss, data, opt; n_epochs=200, plotting=nothing)

    avg_l = []
    
    for epoch = 1:n_epochs
      for d in data
        gs = gradient(params(nn)) do
          l = loss(d...)
        end
        update!(opt, params(nn), gs)
      end
      if !isnothing(plotting)
        plt = plotting[1]
        anim = plotting[2]
        idx = plotting[3]
        avg_loss(data) = mean(map(d -> loss(d[1],d[2]), data))
        avg_l = vcat(avg_l,avg_loss(data))
        if epoch % plotting[4]==0
          plot!(plt, avg_l, color=idx)
          frame(anim, plt)
        end
      end
    end
    
end
```


    forward_nn (generic function with 1 method)


This wrapper function is used as a subroutine in `forward` below, which returns are an ensemble of fitted neural networks. The animation below shows the training loss for each of them. As we can see the different networks produce different outcomes: their parameters were initialized at different random values. This is how we introduce stochasticity and hence incorporate uncertainty around our estimates.


```julia
using Statistics

function forward(𝓜, data, opt; loss_type=:logitbinarycrossentropy, plot_loss=true, n_epochs=200, plot_every=20) 

    anim = nothing
    if plot_loss
        anim = Animation()
        plt = plot(ylim=(0,1), xlim=(0,n_epochs), legend=false, xlab="Epoch", title="Average (training) loss")
        for i in 1:length(𝓜)
            nn = 𝓜[i]
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=(plt, anim, i, plot_every))
        end
    else
        plt = nothing
        for nn in 𝓜
            loss(x, y) = getfield(Flux.Losses,loss_type)(nn(x), y)
            forward_nn(nn, loss, data, opt, n_epochs=n_epochs, plotting=plt)
        end
    end

    return 𝓜, anim
end;
```


```julia
𝓜, anim = forward(𝓜, data, opt, n_epochs=epochs, plot_every=show_every); # fit the ensemble
gif(anim, "www/models_ensemble_loss.gif", fps=10);
```

![](www/models_ensemble_loss.gif)

Once again it is straight-forward to make the model compatible with the package. Note that for an ensemble model the predicted logits and probabilities are just averages over predictions produced by all $K$ models.


```julia
# Step 1)
struct FittedEnsemble <: Models.AbstractFittedModel
    𝓜::AbstractArray
end

# Step 2)
logits(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.flatten(Flux.stack([nn(X) for nn in 𝑴.𝓜],1)),dims=1)
probs(𝑴::FittedEnsemble, X::AbstractArray) = mean(Flux.flatten(Flux.stack([σ.(nn(X)) for nn in 𝑴.𝓜],1)),dims=1)

𝑴=FittedEnsemble(𝓜);
```

Again we plot the predicted probabilities in the feature domain. As expected the ensemble is more *conservative* because it incorporates uncertainty: the predicted probabilities splash out more than before, especially in regions that are not populated by samples.


```julia
plt = plot_contour(X',y,𝑴);
savefig(plt, "www/models_ensemble_contour.png")
```

![](www/models_ensemble_contour.png)

Finally, we use the `GreedyGenerator` for the counterfactual search. For the same desired threshold $\gamma$ as before, the counterfactual ends up somewhat closer to a cluster of original samples. In other words we end up providing more realisitic albeit likely more costly recourse.


```julia
generator = GreedyGenerator(0.25,20,:logitbinarycrossentropy,nothing)
recourse = generate_counterfactual(generator, x̅, 𝑴, target, γ); # generate recourse
```


```julia
T = size(recourse.path)[1]
X_path = reduce(hcat,recourse.path)
ŷ = CounterfactualExplanations.target_probs(probs(recourse.𝑴, X_path),target)
p1 = plot_contour(X',y,𝑴;clegend=false, title="Deep ensemble")
anim = @animate for t in 1:T
    scatter!(p1, [recourse.path[t][1]], [recourse.path[t][2]], ms=5, color=Int(y̅), label="")
    p2 = plot(1:t, ŷ[1:t], xlim=(0,T), ylim=(0, 1), label="p(y̲=" * string(target) * ")", title="Validity", lc=:black)
    Plots.abline!(p2,0,γ,label="threshold γ", ls=:dash) # decision boundary
    plot(p1,p2,size=(800,400))
end
gif(anim, "www/models_greedy_recourse.gif", fps=5);
```

![](www/models_greedy_recourse.gif)
